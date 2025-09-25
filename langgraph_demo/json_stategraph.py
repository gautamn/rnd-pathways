"""Compile a JSON graph (GraphLoader.GraphSpec) into a LangGraph StateGraph.

This turns the deterministic JSON runner into a real LangGraph workflow at runtime.
Each JSON node becomes a StateGraph node; edges are wired with conditional routing
based on current `vars` and edge labels. Node bodies handle:
- Rendering text/prompt via Jinja2
- Prompting user input when needed (deterministic, no LLM)
- Extracting variables using the same heuristics as GraphRunner
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, TypedDict, Optional
import re

from jinja2 import Template
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .graph_loader import GraphLoader, GraphSpec, Node, Edge
from .config import get_settings


class JSONState(TypedDict):
    messages: Annotated[List, add_messages]
    vars: Dict[str, Any]
    response: str


class JSONStateGraphCompiler:
    def __init__(self, spec: GraphSpec):
        self.spec = spec
        self.settings = get_settings()
        self.router_llm: Optional[ChatOpenAI] = None
        if self.settings.use_llm_routing:
            model = self.settings.router_model_name or self.settings.model_name
            self.router_llm = ChatOpenAI(
                model=model,
                temperature=self.settings.router_temperature,
                api_key=self.settings.openai_api_key,
            )

    # ===== Heuristics (shared with GraphRunner) =====
    def _extract_from_input(self, node: Node, user_text: str, vars: Dict[str, Any]) -> None:
        lower = user_text.lower().strip()
        for spec in node.data.extract_vars:
            if not isinstance(spec, list) or not spec:
                continue
            var_name = str(spec[0]) if len(spec) > 0 else None
            var_type = str(spec[1]).lower() if len(spec) > 1 and spec[1] else "string"
            description = str(spec[2]) if len(spec) > 2 and spec[2] else ""
            if not var_name:
                continue

            options: List[str] = []
            if description:
                for line in description.splitlines():
                    line = line.strip()
                    if line and len(line.split()) <= 4:
                        options.append(line)

            chosen: Optional[str] = None
            if options:
                for opt in options:
                    opt_l = opt.lower()
                    if opt_l.replace(" ", "") in lower.replace(" ", "") or opt_l in lower:
                        chosen = opt
                        break

            if not chosen:
                if var_type in ("string", "text"):
                    chosen = user_text.strip()

            if chosen is not None:
                vars[var_name] = chosen

    def _assign_user_name_from_prompt(self, prompt: str) -> Optional[str]:
        m = re.search(r"assigning it to\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)", prompt, flags=re.I)
        if m:
            return m.group(1).strip()
        m = re.search(r"assign(?:ed|ing)\s+to\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)", prompt, flags=re.I)
        if m:
            return m.group(1).strip()
        return None

    def _deterministic_route(self, current_id: str, vars: Dict[str, Any]) -> Optional[str]:
        edges = self.spec.edges_from.get(current_id, [])
        if not edges:
            return None

        for var_val in vars.values():
            if not isinstance(var_val, str):
                continue
            key = var_val.lower()
            for e in edges:
                lbl = (e.label or "").lower()
                if key.replace(" ", "") in lbl.replace(" ", "") or key in lbl:
                    return e.target

        synonyms = {
            "frontend": ["front", "front-end", "front end", "frontend"],
            "backend": ["back", "back-end", "back end", "backend"],
            "ai": [" ai", "ml", "artificial", "intelligence"],
        }
        for keys in synonyms.values():
            for e in edges:
                lbl = (e.label or "").lower()
                if any(k in f" {lbl}" for k in keys):
                    return e.target

        return edges[0].target

    # ===== Optional LLM router =====
    def _llm_route(self, current_id: str, vars: Dict[str, Any]) -> Optional[str]:
        if not self.router_llm:
            return None
        edges = self.spec.edges_from.get(current_id, [])
        if not edges:
            return None
        options = [e.target for e in edges]
        edge_lines = [f"{e.target}: {e.label or ''}" for e in edges]
        sys_prompt = (
            "You are a strict router. Choose the next node id from the provided options. "
            "Return only a JSON object exactly like {\"next_node_id\": \"<id>\"}."
        )
        user_prompt = (
            f"Current node: {current_id}\n" \
            f"Vars: {vars}\n" \
            f"Outgoing options (target_id: label):\n- " + "\n- ".join(edge_lines) + "\n" \
            "Constraints:\n"
            "- Pick exactly one next_node_id from the options.\n"
            "- Do not invent ids."
        )
        try:
            resp = self.router_llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
            text = (resp.content or "").strip()
            # Basic JSON extraction without strict parsing (avoid dependency bloat)
            m = re.search(r'\{\s*"next_node_id"\s*:\s*"([^"]+)"\s*\}', text)
            if not m:
                return None
            nxt = m.group(1)
            return nxt if nxt in options else None
        except Exception:
            return None

    # ===== Node factory =====
    def _make_node_fn(self, node_id: str):
        node = self.spec.nodes[node_id]

        def node_fn(state: JSONState) -> Dict[str, Any]:
            vars = state.get("vars", {})
            raw_text = node.data.text or node.data.prompt
            if raw_text:
                try:
                    text = Template(raw_text).render(**vars)
                except Exception:
                    text = raw_text
                # Print prompt/output deterministically
                print(f"\nAI: {text}\n")
                state["response"] = text

            expects_input = node.data.is_start or (not node.data.skip_user_response)
            if expects_input and not node.data.type.lower().startswith("end"):
                user_text = input("You: ")
                self._extract_from_input(node, user_text, vars)

            if node.data.prompt and not node.data.is_start:
                name = self._assign_user_name_from_prompt(node.data.prompt)
                if name:
                    vars["user_name"] = name

            update = {"vars": vars}
            if state.get("response"):
                update["messages"] = [AIMessage(content=state["response"])]
            return update

        return node_fn

    # ===== Router factory =====
    def _make_router_fn(self, node_id: str):
        def router(state: JSONState) -> str:
            # Try LLM router first if enabled, otherwise deterministic.
            nxt = None
            if self.router_llm is not None:
                nxt = self._llm_route(node_id, state.get("vars", {}))
            if nxt is None:
                nxt = self._deterministic_route(node_id, state.get("vars", {}))
            return nxt if nxt is not None else END
        return router

    # ===== Compile to StateGraph =====
    def compile(self):
        graph = StateGraph(JSONState)

        # Add nodes for each JSON node id
        for nid in self.spec.nodes.keys():
            graph.add_node(nid, self._make_node_fn(nid))

        # Set entry
        graph.set_entry_point(self.spec.start_id)

        # For each node, add conditional edges based on router
        for nid, node in self.spec.nodes.items():
            if node.data.type.lower().startswith("end"):
                graph.add_edge(nid, END)
            else:
                graph.add_conditional_edges(nid, self._make_router_fn(nid))

        return graph.compile()


def run_json_stategraph(path: str) -> None:
    spec = GraphLoader.load(path)
    compiler = JSONStateGraphCompiler(spec)
    app = compiler.compile()
    # Initial empty state
    state: JSONState = {"messages": [], "vars": {}, "response": ""}
    # Invoke; the graph proceeds until END
    app.invoke(state)
