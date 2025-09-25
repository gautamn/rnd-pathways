"""General-purpose deterministic JSON graph loader and runner.

This module loads a simple node/edge JSON graph (like sample-graph.json)
and runs a deterministic, prompt-driven workflow. It does not require
LLM calls for routing and can operate without LangGraph for interactivity.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Template


@dataclass
class NodeData:
    name: str
    text: Optional[str] = None
    prompt: Optional[str] = None
    is_start: bool = False
    type: str = "Default"
    extract_vars: List[List[Any]] = field(default_factory=list)
    model_options: Dict[str, Any] = field(default_factory=dict)
    skip_user_response: bool = False


@dataclass
class Node:
    id: str
    data: NodeData


@dataclass
class Edge:
    source: str
    target: str
    label: str = ""


@dataclass
class GraphSpec:
    nodes: Dict[str, Node]
    edges_from: Dict[str, List[Edge]]
    start_id: str
    end_ids: List[str]


class GraphLoader:
    """Loads the JSON graph into a convenient in-memory structure."""

    @staticmethod
    def load(path: str | Path) -> GraphSpec:
        p = Path(path)
        payload = json.loads(p.read_text())

        raw_nodes: List[Dict[str, Any]] = payload.get("nodes", [])
        raw_edges: List[Dict[str, Any]] = payload.get("edges", [])

        nodes: Dict[str, Node] = {}
        start_id: Optional[str] = None
        end_ids: List[str] = []

        for rn in raw_nodes:
            # Some graphs include non-node objects (like globalConfig). Skip those.
            if "id" not in rn or "data" not in rn:
                continue
            nid = rn["id"]
            d = rn.get("data", {}) or {}
            node_type = rn.get("type", "Default")
            model_opts = d.get("modelOptions", {}) or {}
            data = NodeData(
                name=d.get("name") or d.get("text") or d.get("prompt") or nid,
                text=d.get("text"),
                prompt=d.get("prompt"),
                is_start=bool(d.get("isStart", False)),
                type=node_type,
                extract_vars=d.get("extractVars", []) or [],
                model_options=model_opts,
                skip_user_response=bool(model_opts.get("skipUserResponse", False)),
            )
            nodes[nid] = Node(id=nid, data=data)
            if data.is_start:
                start_id = nid
            if node_type.lower().startswith("end"):
                end_ids.append(nid)

        if not nodes:
            raise ValueError("No valid nodes found in graph JSON.")
        if not start_id:
            raise ValueError("No start node (isStart=true) found in graph JSON.")
        if not end_ids:
            # Allow graphs without explicit end node; runner will stop after no outgoing edges
            end_ids = []

        edges_from: Dict[str, List[Edge]] = {}
        for re_ in raw_edges:
            src = re_.get("source")
            tgt = re_.get("target")
            if not src or not tgt:
                continue
            label = (re_.get("data", {}) or {}).get("label", "")
            edges_from.setdefault(src, []).append(
                Edge(source=src, target=tgt, label=label)
            )

        return GraphSpec(nodes=nodes, edges_from=edges_from, start_id=start_id, end_ids=end_ids)


class GraphRunner:
    """Deterministic interactive runner for the loaded graph."""

    def __init__(self, spec: GraphSpec):
        self.spec = spec
        self.vars: Dict[str, Any] = {}

    def _extract_from_input(self, node: Node, user_text: str) -> None:
        """Extract variables based on node.data.extract_vars definitions.
        Expected format per var: [var_name, type, description, required_bool]

        Generic heuristic:
        - If description lists options line-by-line, try to match keywords from user_text.
        - Otherwise, store the raw user_text for string types.
        """
        lower = user_text.lower().strip()
        for spec in node.data.extract_vars:
            if not isinstance(spec, list) or not spec:
                continue
            var_name = str(spec[0]) if len(spec) > 0 else None
            var_type = str(spec[1]).lower() if len(spec) > 1 and spec[1] else "string"
            description = str(spec[2]) if len(spec) > 2 and spec[2] else ""
            # required_flag = bool(spec[3]) if len(spec) > 3 else False

            if not var_name:
                continue

            # Try to infer options from description (each on a separate line)
            options: List[str] = []
            if description:
                for line in description.splitlines():
                    line = line.strip()
                    # Skip sentences, collect single tokens/labels
                    if line and len(line.split()) <= 4:
                        options.append(line)

            chosen: Optional[str] = None
            if options:
                # Match any option keyword in user input
                for opt in options:
                    opt_l = opt.lower()
                    # Allow partial matches (e.g., frontend -> front end)
                    if opt_l.replace(" ", "") in lower.replace(" ", "") or opt_l in lower:
                        chosen = opt
                        break

            if not chosen:
                # Fallback: assign raw user text for strings; leave untouched otherwise
                if var_type in ("string", "text"):
                    chosen = user_text.strip()

            if chosen is not None:
                self.vars[var_name] = chosen

    def _deterministic_route(self, current_id: str) -> Optional[str]:
        """Choose next node deterministically based on vars and edge labels."""
        edges = self.spec.edges_from.get(current_id, [])
        if not edges:
            return None

        # Try to match any current variable values against edge labels
        for var_val in self.vars.values():
            if not isinstance(var_val, str):
                continue
            key = var_val.lower()
            for e in edges:
                lbl = (e.label or "").lower()
                # Loose fuzzy match
                if key.replace(" ", "") in lbl.replace(" ", "") or key in lbl:
                    return e.target

        # Special-case synonyms for common intents (frontend/backend/ai)
        synonyms = {
            "frontend": ["front", "front-end", "front end", "frontend"],
            "backend": ["back", "back-end", "back end", "backend"],
            "ai": [" ai", "ml", "artificial", "intelligence"],
        }
        for name, keys in synonyms.items():
            for e in edges:
                lbl = (e.label or "").lower()
                if any(k in f" {lbl}" for k in keys):
                    return e.target

        # Fallback to first edge
        return edges[0].target

    def _assign_user_name_from_prompt(self, prompt: str) -> Optional[str]:
        """Heuristic to capture the assignee name from the node prompt."""
        # Look for 'assigning it to <Name>'
        m = re.search(r"assigning it to\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)", prompt, flags=re.I)
        if m:
            return m.group(1).strip()
        # Or 'assigning it to <Name>.'
        m = re.search(r"assign(?:ed|ing)\s+to\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)", prompt, flags=re.I)
        if m:
            return m.group(1).strip()
        return None

    def run_interactive(self) -> None:
        """Run the graph interactively in the terminal."""
        current_id = self.spec.start_id

        while True:
            node = self.spec.nodes[current_id]
            nd = node.data

            # Display node text or prompt (render variables if present)
            raw_text = nd.text or nd.prompt
            if raw_text:
                try:
                    text = Template(raw_text).render(**self.vars)
                except Exception:
                    text = raw_text
                if nd.type.lower().startswith("end"):
                    print(f"\nAI: {text}\n")
                    break
                else:
                    print(f"\nAI: {text}\n")

            # If this node expects user input, collect it and extract vars.
            expects_input = nd.is_start or (not nd.skip_user_response)
            # For "system" or auto nodes, authors may set skipUserResponse=true.
            if expects_input and not nd.type.lower().startswith("end"):
                user_text = input("You: ")
                self._extract_from_input(node, user_text)

            # For assignment nodes, try to set user_name based on prompt
            if nd.prompt and not nd.is_start:
                name = self._assign_user_name_from_prompt(nd.prompt)
                if name:
                    self.vars["user_name"] = name

            # Decide next node deterministically
            next_id = self._deterministic_route(current_id)
            if not next_id:
                # No outgoing edge -> stop
                break
            current_id = next_id
