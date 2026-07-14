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
        'mish': lambda x: x * np.tanh(np.log(1 + np.exp(np.clip(x, -20, 20)))),
        'elu': lambda x: np.where(x > 0, x, 1.0 * (np.exp(x) - 1)),
        'selu': lambda x: 1.0507 * np.where(x > 0, x, 1.6733 * (np.exp(x) - 1)),
        'sin': np.sin,
        'cos': np.cos,
        'gaussian': lambda x: np.exp(-x**2),
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
        self.action_entropy = 0.0  # Entropy of actions taken during episode
        
        # Meta-mutation: each genome has its own mutation rates
        self.meta_rates = {
            'weight_mutate_power': 0.5,
            'conn_add_prob': 0.2,
            'node_add_prob': 0.2,
        }
        
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
    
    def _compute_fan_in(self, node_id: int) -> int:
        """Compute fan-in (number of incoming connections) for a node."""
        return sum(1 for c in self.connections.values() 
                  if c.enabled and c.out_node == node_id)
    
    def _kaiming_weight(self, out_node: int) -> float:
        """Generate a weight using Kaiming/He initialization."""
        fan_in = self._compute_fan_in(out_node)
        if fan_in == 0:
            # First connection to this node - use small random weight
            return random.gauss(0, 0.1)
        # He initialization: N(0, sqrt(2/fan_in))
        std = np.sqrt(2.0 / fan_in)
        return random.gauss(0, std)
    
    def add_connection(self, in_node: int, out_node: int, 
                       weight: Optional[float] = None) -> bool:
        """Add a new connection between two nodes."""
        if (in_node, out_node) in self.connections:
            return False
        if in_node not in self.nodes or out_node not in self.nodes:
            return False
        if weight is None:
            weight = self._kaiming_weight(out_node)
        
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
    
    def mutate_add_skip(self, add_prob: float = 0.05):
        """Add a skip connection (ResNet-style) between non-consecutive layers."""
        if random.random() >= add_prob:
            return
        
        # Find all pairs of nodes where we can add a skip connection
        # (nodes that don't already have a direct connection)
        candidates = []
        for i, n1 in self.nodes.items():
            for j, n2 in self.nodes.items():
                if i == j:
                    continue
                if n1.node_type == 'output' or n2.node_type == 'input':
                    continue
                if (i, j) in self.connections:
                    continue
                # Check if there's already a path (don't skip too far)
                candidates.append((i, j))
        
        if candidates:
            in_id, out_id = random.choice(candidates)
            # Skip connections use small initial weights
            self.add_connection(in_id, out_id, random.gauss(0, 0.01))
    
    def gradient_update(self, eval_func, step_size: float = 0.01, n_samples: int = 4):
        """
        Perform a gradient-like weight update using finite differences.
        Evaluates n_samples perturbed versions and moves weights toward
        the direction of improvement.
        """
        # Get baseline fitness
        baseline = eval_func(self)
        
        # Collect enabled connections
        enabled_conns = [(k, c) for k, c in self.connections.items() if c.enabled]
        if not enabled_conns:
            return
        
        # Sample a subset of connections to update
        n_update = max(1, len(enabled_conns) // 4)
        to_update = random.sample(enabled_conns, min(n_update, len(enabled_conns)))
        
        for key, conn in to_update:
            # Estimate gradient via finite differences
            improvements = []
            for _ in range(n_samples):
                delta = random.gauss(0, step_size)
                conn.weight += delta
                fit = eval_func(self)
                conn.weight -= delta  # Restore
                improvements.append((fit - baseline) / delta if delta != 0 else 0)
            
            # Update weight in direction of improvement
            grad = np.mean(improvements)
            conn.weight += step_size * grad
            conn.weight = np.clip(conn.weight, -30, 30)
    
    def mutate_meta(self, meta_rate: float = 0.1):
        """Mutate this genome's own mutation rates (meta-mutation)."""
        for key in self.meta_rates:
            if random.random() < meta_rate:
                self.meta_rates[key] *= random.gauss(1.0, 0.2)
                self.meta_rates[key] = max(0.01, min(1.0, self.meta_rates[key]))
    
    def mutate(self, config: dict):
        """Apply all mutation operators using genome-specific rates."""
        # Override global config with genome-specific meta-rates
        weight_power = self.meta_rates.get('weight_mutate_power', config.get('weight_mutate_power', 0.5))
        conn_add = self.meta_rates.get('conn_add_prob', config.get('conn_add_prob', 0.2))
        node_add = self.meta_rates.get('node_add_prob', config.get('node_add_prob', 0.2))
        
        self.mutate_weight(
            config.get('weight_mutate_prob', 0.9),
            weight_power,
            config.get('weight_replace_prob', 0.1)
        )
        self.mutate_bias(
            config.get('bias_mutate_prob', 0.7),
            config.get('bias_mutate_power', 0.5)
        )
        self.mutate_activation(config.get('activation_mutate_rate', 0.1))
        self.mutate_add_connection(conn_add)
        self.mutate_add_node(node_add)
        self.mutate_add_skip(config.get('skip_add_prob', 0.05))
        self.mutate_toggle_connection(config.get('toggle_prob', 0.01))
        self.mutate_remove_connection(config.get('conn_remove_prob', 0.1))
        self.mutate_remove_node(config.get('node_remove_prob', 0.05))
        
        # Also mutate the mutation rates themselves
        self.mutate_meta(config.get('meta_mutate_rate', 0.1))
    
    def compatibility_distance(self, other: 'Genome') -> float:
        """Compute NEAT-style compatibility distance."""
        if not self.connections or not other.connections:
            return 100.0
        
        # Count matching genes
        matching = 0
        disjoint = 0
        weight_diff = 0.0
        
        self_innovs = {c.innov: c for c in self.connections.values()}
        other_innovs = {c.innov: c for c in other.connections.values()}
        
        all_innovs = set(self_innovs.keys()) | set(other_innovs.keys())
        max_innov = max(all_innovs) if all_innovs else 1
        
        for innov in all_innovs:
            if innov in self_innovs and innov in other_innovs:
                matching += 1
                weight_diff += abs(self_innovs[innov].weight - other_innovs[innov].weight)
            else:
                disjoint += 1
        
        if matching == 0:
            return disjoint + weight_diff
        
        # NEAT compatibility: c1 * disjoint/N + c2 * weight_diff/matching
        n = max(len(self.connections), len(other.connections))
        n = max(n, 1)
        return (1.0 * disjoint / n) + (0.5 * weight_diff / matching)
    
    def crossover(self, other: 'Genome') -> 'Genome':
        """
        Crossover with another genome using historical markings (NEAT-style).
        Returns a new child genome.
        """
        child = Genome(
            self.key * 10000 + other.key,
            self.num_inputs,
            self.num_outputs
        )
        child.nodes = {}
        child.connections = {}
        
        # Determine more fit parent
        self_adj = getattr(self, 'adjusted_fitness', self.fitness) or -9999
        other_adj = getattr(other, 'adjusted_fitness', other.fitness) or -9999
        if other_adj > self_adj:
            primary, secondary = other, self
        else:
            primary, secondary = self, other
        
        # Merge nodes: all nodes from both parents
        all_node_ids = set(primary.nodes.keys()) | set(secondary.nodes.keys())
        for nid in all_node_ids:
            if nid in primary.nodes and nid in secondary.nodes:
                child.nodes[nid] = copy.deepcopy(
                    primary.nodes[nid] if random.random() < 0.5 else secondary.nodes[nid]
                )
            elif nid in primary.nodes:
                child.nodes[nid] = copy.deepcopy(primary.nodes[nid])
            else:
                child.nodes[nid] = copy.deepcopy(secondary.nodes[nid])
        
        # Merge connections by innovation number (NEAT-style)
        primary_innovs = {c.innov: c for c in primary.connections.values()}
        secondary_innovs = {c.innov: c for c in secondary.connections.values()}
        all_innovs = set(primary_innovs.keys()) | set(secondary_innovs.keys())
        
        for innov in all_innovs:
            if innov in primary_innovs and innov in secondary_innovs:
                # Matching gene - randomly choose from either parent
                child.connections[(primary_innovs[innov].in_node, primary_innovs[innov].out_node)] = \
                    copy.deepcopy(primary_innovs[innov] if random.random() < 0.5 else secondary_innovs[innov])
            elif innov in primary_innovs:
                # Disjoint/excess - inherit from more fit parent
                conn = primary_innovs[innov]
                child.connections[(conn.in_node, conn.out_node)] = copy.deepcopy(conn)
            else:
                # Only in secondary
                conn = secondary_innovs[innov]
                child.connections[(conn.in_node, conn.out_node)] = copy.deepcopy(conn)
        
        return child
    
    def get_characterization(self) -> Tuple[float, float, float, float, float, float]:
        """
        Get the 6D behavioral-topological characterization.
        Returns (depth, hidden_ratio, conn_ratio, activation_diversity, weight_sparsity, action_entropy)
        Each dimension is normalized to [0, 1].
        """
        input_ids = set(n.id for n in self.nodes.values() if n.node_type == 'input')
        output_ids = set(n.id for n in self.nodes.values() if n.node_type == 'output')
        hidden_nodes = [n for n in self.nodes.values() if n.node_type == 'hidden']
        n_enabled = sum(1 for c in self.connections.values() if c.enabled)
        
        # Dimension 1: Network depth (longest path from any input to any output)
        depth = self._compute_depth(input_ids, output_ids)
        max_depth = 20  # Upper bound
        depth_norm = min(depth / max_depth, 1.0)
        
        # Dimension 2: Hidden node ratio
        max_hidden = 50
        hidden_ratio = min(len(hidden_nodes) / max_hidden, 1.0)
        
        # Dimension 3: Connection ratio (enabled / possible excluding self-loops)
        n_nodes = len(self.nodes)
        max_possible = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
        conn_ratio = min(n_enabled / max_possible, 1.0) if max_possible > 0 else 0
        
        # Dimension 4: Activation diversity (normalized entropy)
        activations = [n.activation for n in self.nodes.values() if n.node_type != 'input']
        if not activations:
            act_diversity = 0.0
        else:
            counts = {}
            for a in activations:
                counts[a] = counts.get(a, 0) + 1
            probs = [c / len(activations) for c in counts.values()]
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            act_diversity = min(entropy / np.log(len(NodeGene.AVAILABLE_ACTIVATIONS)), 1.0)
        
        # Dimension 5: Weight sparsity (fraction of |weight| < 0.1)
        enabled_weights = [abs(c.weight) for c in self.connections.values() if c.enabled]
        if enabled_weights:
            sparsity = sum(1 for w in enabled_weights if w < 0.1) / len(enabled_weights)
        else:
            sparsity = 0.0
        
        # Dimension 6: Action entropy (behavioral diversity during episode)
        # Normalize: high entropy = exploring, low entropy = exploiting
        act_ent = min(self.action_entropy / 2.0, 1.0)
        
        return (depth_norm, hidden_ratio, conn_ratio, act_diversity, sparsity, act_ent)
    
    def _compute_depth(self, input_ids: set, output_ids: set) -> int:
        """Compute longest path from any input to any output."""
        if not input_ids or not output_ids:
            return 0
        
        # Build adjacency list of enabled connections
        adj = {nid: [] for nid in self.nodes}
        for conn in self.connections.values():
            if conn.enabled:
                adj[conn.in_node].append(conn.out_node)
        
        # BFS from each input to find max depth to any output
        max_depth = 0
        for start in input_ids:
            visited = {start: 0}
            queue = [start]
            while queue:
                node = queue.pop(0)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited[neighbor] = visited[node] + 1
                        queue.append(neighbor)
                        if neighbor in output_ids:
                            max_depth = max(max_depth, visited[neighbor])
        
        return max_depth
    
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
