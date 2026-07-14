"""
Atlas-NEAT Genome representation.
Similar to NEAT but with modern improvements:
- Multiple activation functions per node
- Optional recurrent connections
- More efficient encoding
"""

import random
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import numpy as np


@dataclass 
class NodeGene:
    """A node in the neural network."""
    id: int
    node_type: str  # 'input', 'hidden', 'output'
    activation: str = 'tanh'
    bias: float = 0.0
    response: float = 1.0
    
    # Supported activations with their functions
    ACTIVATIONS = {
        'tanh': np.tanh,
        'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
        'relu': lambda x: np.maximum(0, x),
        'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
        'swish': lambda x: x / (1.0 + np.exp(-np.clip(x, -500, 500))),
        'gelu': lambda x: 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3))),
        'sin': np.sin,
        'cos': np.cos,
        'abs': np.abs,
        'identity': lambda x: x,
    }
    
    AVAILABLE_ACTIVATIONS = list(ACTIVATIONS.keys())
    
    def activate(self, x):
        """Apply activation function."""
        return self.ACTIVATIONS.get(self.activation, np.tanh)(x)


@dataclass
class ConnectionGene:
    """A connection between two nodes."""
    in_node: int
    out_node: int
    weight: float = 0.0
    enabled: bool = True
    innov: int = 0  # Innovation number


class Genome:
    """
    Atlas-NEAT Genome.
    Encodes a neural network as a set of nodes and connections.
    """
    
    _innovation_counter = 0
    _innovation_history = {}  # (in_node, out_node) -> innov_number
    
    def __init__(self, key: int, num_inputs: int, num_outputs: int, 
                 seed_activations: Optional[List[str]] = None):
        self.key = key
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[Tuple[int, int], ConnectionGene] = {}
        self.fitness: Optional[float] = None
        
        # Behavioral stats (computed during evaluation)
        self.activation_sparsity = 0.0
        self.output_entropy = 0.0
        
        # Create input nodes
        for i in range(num_inputs):
            self.nodes[i] = NodeGene(i, 'input', 'identity')
        
        # Create output nodes
        for i in range(num_outputs):
            node_id = num_inputs + i
            act = 'tanh'
            if seed_activations and i < len(seed_activations):
                act = seed_activations[i]
            self.nodes[node_id] = NodeGene(node_id, 'output', act)
        
        # Create initial connections (fully connected)
        for i in range(num_inputs):
            for j in range(num_outputs):
                out_id = num_inputs + j
                self.add_connection(i, out_id, random.gauss(0, 1))
    
    @classmethod
    def reset_innovation(cls):
        """Reset the global innovation counter."""
        cls._innovation_counter = 0
        cls._innovation_history = {}
    
    @classmethod
    def get_innovation_number(cls, in_node: int, out_node: int) -> int:
        """Get or create innovation number for a connection."""
        key = (in_node, out_node)
        if key not in cls._innovation_history:
            cls._innovation_history[key] = cls._innovation_counter
            cls._innovation_counter += 1
        return cls._innovation_history[key]
    
    def add_connection(self, in_node: int, out_node: int, 
                       weight: Optional[float] = None) -> bool:
        """Add a new connection between two nodes."""
        if (in_node, out_node) in self.connections:
            return False
        if in_node not in self.nodes or out_node not in self.nodes:
            return False
        if weight is None:
            weight = random.gauss(0, 1)
        
        innov = self.get_innovation_number(in_node, out_node)
        self.connections[(in_node, out_node)] = ConnectionGene(
            in_node, out_node, weight, True, innov
        )
        return True
    
    def add_node(self, connection_key: Tuple[int, int], 
                 activation: Optional[str] = None) -> Optional[int]:
        """
        Add a new node by splitting an existing connection.
        Returns the new node ID or None if failed.
        """
        if connection_key not in self.connections:
            return None
        
        conn = self.connections[connection_key]
        if not conn.enabled:
            return None
        
        # Disable old connection
        conn.enabled = False
        
        # Create new node
        new_id = max(self.nodes.keys()) + 1
        if activation is None:
            activation = random.choice(NodeGene.AVAILABLE_ACTIVATIONS)
        
        self.nodes[new_id] = NodeGene(new_id, 'hidden', activation)
        
        # Add two new connections
        self.add_connection(conn.in_node, new_id, 1.0)
        self.add_connection(new_id, conn.out_node, conn.weight)
        
        return new_id
    
    def mutate_weight(self, perturb_prob: float = 0.9, 
                      perturb_power: float = 0.5,
                      replace_prob: float = 0.1):
        """Mutate connection weights."""
        for conn in self.connections.values():
            if not conn.enabled:
                continue
            r = random.random()
            if r < perturb_prob:
                conn.weight += random.gauss(0, perturb_power)
            elif r < perturb_prob + replace_prob:
                conn.weight = random.gauss(0, 1)
            # Clip
            conn.weight = np.clip(conn.weight, -30, 30)
    
    def mutate_bias(self, perturb_prob: float = 0.7,
                    perturb_power: float = 0.5):
        """Mutate node biases."""
        for node in self.nodes.values():
            if node.node_type == 'input':
                continue
            if random.random() < perturb_prob:
                node.bias += random.gauss(0, perturb_power)
                node.bias = np.clip(node.bias, -30, 30)
    
    def mutate_activation(self, mutate_rate: float = 0.1):
        """Mutate node activation functions."""
        for node in self.nodes.values():
            if node.node_type == 'input':
                continue
            if random.random() < mutate_rate:
                node.activation = random.choice(NodeGene.AVAILABLE_ACTIVATIONS)
    
    def mutate_add_connection(self, add_prob: float = 0.2):
        """Add a new random connection."""
        if random.random() >= add_prob:
            return
        
        # Find all possible connections (feedforward only)
        possible = []
        for in_id in self.nodes:
            for out_id in self.nodes:
                if in_id == out_id:
                    continue
                if (in_id, out_id) in self.connections:
                    continue
                in_node = self.nodes[in_id]
                out_node = self.nodes[out_id]
                # Feedforward: inputs connect to hidden/output, hidden connects to hidden/output
                if in_node.node_type != 'output' and out_node.node_type != 'input':
                    possible.append((in_id, out_id))
        
        if possible:
            in_id, out_id = random.choice(possible)
            self.add_connection(in_id, out_id)
    
    def mutate_add_node(self, add_prob: float = 0.2):
        """Add a new node by splitting an existing connection."""
        if random.random() >= add_prob:
            return
        
        enabled_conns = [k for k, c in self.connections.items() if c.enabled]
        if enabled_conns:
            conn_key = random.choice(enabled_conns)
            self.add_node(conn_key)
    
    def mutate_toggle_connection(self, toggle_prob: float = 0.01):
        """Toggle connection enabled/disabled."""
        for conn in self.connections.values():
            if random.random() < toggle_prob:
                conn.enabled = not conn.enabled
    
    def mutate_remove_connection(self, remove_prob: float = 0.1):
        """Remove a random connection."""
        if random.random() >= remove_prob:
            return
        if len(self.connections) <= self.num_inputs * self.num_outputs:
            return  # Keep at least initial connections
        
        # Don't remove essential connections
        removable = [k for k in self.connections.keys()]
        if removable:
            key = random.choice(removable)
            del self.connections[key]
    
    def mutate_remove_node(self, remove_prob: float = 0.05):
        """Remove a random hidden node."""
        if random.random() >= remove_prob:
            return
        
        hidden = [nid for nid, n in self.nodes.items() if n.node_type == 'hidden']
        if not hidden:
            return
        
        node_id = random.choice(hidden)
        self._remove_node(node_id)
    
    def _remove_node(self, node_id: int):
        """Remove a node and all its connections, preserving connectivity."""
        if node_id not in self.nodes:
            return
        
        # Get connections to/from this node
        in_conns = [(k, c) for k, c in self.connections.items() if k[1] == node_id and c.enabled]
        out_conns = [(k, c) for k, c in self.connections.items() if k[0] == node_id and c.enabled]
        
        # Try to create bypass connections (from inputs to outputs of removed node)
        for in_key, in_conn in in_conns:
            for out_key, out_conn in out_conns:
                # Skip if would create self-loop
                if in_conn.in_node == out_conn.out_node:
                    continue
                # Add bypass if doesn't exist
                if (in_conn.in_node, out_conn.out_node) not in self.connections:
                    self.add_connection(in_conn.in_node, out_conn.out_node, 
                                       in_conn.weight * out_conn.weight)
        
        # Remove all connections to/from this node
        to_remove = [k for k in self.connections.keys() 
                     if k[0] == node_id or k[1] == node_id]
        for k in to_remove:
            del self.connections[k]
        
        del self.nodes[node_id]
    
    def mutate(self, config: dict):
        """Apply all mutation operators."""
        self.mutate_weight(
            config.get('weight_mutate_prob', 0.9),
            config.get('weight_mutate_power', 0.5),
            config.get('weight_replace_prob', 0.1)
        )
        self.mutate_bias(
            config.get('bias_mutate_prob', 0.7),
            config.get('bias_mutate_power', 0.5)
        )
        self.mutate_activation(config.get('activation_mutate_rate', 0.1))
        self.mutate_add_connection(config.get('conn_add_prob', 0.2))
        self.mutate_add_node(config.get('node_add_prob', 0.2))
        self.mutate_toggle_connection(config.get('toggle_prob', 0.01))
        self.mutate_remove_connection(config.get('conn_remove_prob', 0.1))
        self.mutate_remove_node(config.get('node_remove_prob', 0.05))
    
    def crossover(self, other: 'Genome') -> 'Genome':
        """
        Crossover with another genome using historical markings.
        Returns a new child genome.
        """
        child = Genome(
            self.key * 10000 + other.key,
            self.num_inputs,
            self.num_outputs
        )
        child.nodes = {}
        child.connections = {}
        
        # Inherit nodes from more fit parent
        if (other.fitness or -9999) > (self.fitness or -9999):
            primary, secondary = other, self
        else:
            primary, secondary = self, other
        
        # Merge nodes
        all_node_ids = set(primary.nodes.keys()) | set(secondary.nodes.keys())
        for nid in all_node_ids:
            if nid in primary.nodes and nid in secondary.nodes:
                # Matching - randomly choose
                child.nodes[nid] = copy.deepcopy(
                    primary.nodes[nid] if random.random() < 0.5 else secondary.nodes[nid]
                )
            elif nid in primary.nodes:
                # Disjoint/excess from primary
                child.nodes[nid] = copy.deepcopy(primary.nodes[nid])
            else:
                # Disjoint/excess from secondary
                child.nodes[nid] = copy.deepcopy(secondary.nodes[nid])
        
        # Merge connections
        all_conn_keys = set(primary.connections.keys()) | set(secondary.connections.keys())
        for ck in all_conn_keys:
            if ck in primary.connections and ck in secondary.connections:
                # Matching - randomly choose
                child.connections[ck] = copy.deepcopy(
                    primary.connections[ck] if random.random() < 0.5 else secondary.connections[ck]
                )
            elif ck in primary.connections:
                child.connections[ck] = copy.deepcopy(primary.connections[ck])
            else:
                child.connections[ck] = copy.deepcopy(secondary.connections[ck])
        
        return child
    
    def get_characterization(self) -> Tuple[float, float, float]:
        """
        Get the 3D behavioral-topological characterization.
        Returns (complexity, connectivity, activation_diversity)
        """
        n_nodes = len([n for n in self.nodes.values() if n.node_type != 'input'])
        n_total = len(self.nodes)
        n_enabled = sum(1 for c in self.connections.values() if c.enabled)
        n_possible = n_total * (n_total - 1) / 2 if n_total > 1 else 1
        
        # Dimension 1: Complexity (normalized node count)
        max_nodes = 100  # Reasonable upper bound
        complexity = min(n_nodes / max_nodes, 1.0)
        
        # Dimension 2: Connectivity density
        connectivity = min(n_enabled / max(n_possible, 1), 1.0)
        
        # Dimension 3: Activation diversity (entropy of activation types)
        activations = [n.activation for n in self.nodes.values() if n.node_type != 'input']
        if not activations:
            diversity = 0.0
        else:
            counts = {}
            for a in activations:
                counts[a] = counts.get(a, 0) + 1
            probs = [c / len(activations) for c in counts.values()]
            diversity = -sum(p * np.log(p + 1e-10) for p in probs) / np.log(len(NodeGene.AVAILABLE_ACTIVATIONS))
        
        return (complexity, connectivity, diversity)
    
    def get_complexity_score(self) -> int:
        """Get total network complexity (nodes + enabled connections)."""
        return len(self.nodes) + sum(1 for c in self.connections.values() if c.enabled)
    
    def create_network(self):
        """Create a feedforward network from this genome."""
        return FeedForwardNetwork(self)


class FeedForwardNetwork:
    """Feedforward network built from a genome."""
    
    def __init__(self, genome: Genome):
        self.genome = genome
        self.input_nodes = [n.id for n in genome.nodes.values() if n.node_type == 'input']
        self.output_nodes = [n.id for n in genome.nodes.values() if n.node_type == 'output']
        self.hidden_nodes = [n.id for n in genome.nodes.values() if n.node_type == 'hidden']
        
        # Compute evaluation order (topological sort)
        self.eval_order = self._compute_eval_order()
    
    def _compute_eval_order(self) -> List[int]:
        """Compute topological ordering of nodes."""
        # Build adjacency list of enabled connections
        incoming = {nid: [] for nid in self.genome.nodes}
        for conn in self.genome.connections.values():
            if conn.enabled:
                incoming[conn.out_node].append(conn.in_node)
        
        # Topological sort
        order = list(self.input_nodes)
        visited = set(order)
        
        # Iteratively add nodes whose predecessors are all visited
        changed = True
        while changed:
            changed = False
            for nid in self.genome.nodes:
                if nid in visited:
                    continue
                preds = incoming[nid]
                if all(p in visited for p in preds):
                    order.append(nid)
                    visited.add(nid)
                    changed = True
        
        return order
    
    def activate(self, inputs: np.ndarray) -> List[float]:
        """Run the network forward."""
        values = {}
        
        # Set input values
        for i, nid in enumerate(sorted(self.input_nodes)):
            values[nid] = float(inputs[i]) if i < len(inputs) else 0.0
        
        # Evaluate in topological order
        for nid in self.eval_order:
            if nid in values:
                continue  # Already set (input)
            
            node = self.genome.nodes[nid]
            
            # Sum weighted inputs
            total = node.bias
            for conn in self.genome.connections.values():
                if conn.enabled and conn.out_node == nid:
                    total += conn.weight * values.get(conn.in_node, 0)
            
            values[nid] = node.activate(total)
        
        # Return outputs - use 0 for disconnected outputs
        return [values.get(nid, 0.0) for nid in sorted(self.output_nodes)]


def create_random_genome(key: int, num_inputs: int, num_outputs: int) -> Genome:
    """Create a random minimal genome."""
    return Genome(key, num_inputs, num_outputs)
