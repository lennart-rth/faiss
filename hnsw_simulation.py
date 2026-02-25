import numpy as np
import heapq
import math
import matplotlib.pyplot as plt

# ==========================================
# 1. FAISS UTILITIES (RANDOM & HEAPS)
# ==========================================

class FaissRandomGenerator:
    """Mimics faiss::RandomGenerator (Simple LCG used in Faiss)"""
    def __init__(self, seed=12345):
        self.state = seed
    
    def rand_int(self, max_val):
        self.state = (self.state * 1103515245 + 12345) & 0x7fffffff
        return self.state % max_val

    def rand_float(self):
        self.state = (self.state * 1103515245 + 12345) & 0x7fffffff
        return (self.state & 0x7fffffff) / 2147483648.0

# ==========================================
# 2. CORE HNSW IMPLEMENTATION
# ==========================================
class FaissHNSW:
    def __init__(self, dim, M, efConstruction=40, efSearch=16):
        self.d = dim
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.rng = FaissRandomGenerator()
        
        self.assign_probas = []
        self.cum_nneighbor_per_level = [0]
        level_mult = 1.0 / math.log(M)
        
        nn = 0
        for level in range(100):
            proba = math.exp(-level / level_mult) * (1 - math.exp(-1 / level_mult))
            if proba < 1e-9: break
            self.assign_probas.append(proba)
            nn += (M * 2) if level == 0 else M
            self.cum_nneighbor_per_level.append(nn)

        self.data = []           
        self.levels = []         
        self.offsets = [0]       
        self.neighbors = []      
        self.entry_point = -1
        self.max_level = -1

    def _get_dist(self, i1, i2=None, vec=None):
        v1 = self.data[i1]
        v2 = vec if vec is not None else self.data[i2]
        return np.linalg.norm(v1 - v2)

    def _random_level(self):
        f = self.rng.rand_float()
        for level, proba in enumerate(self.assign_probas):
            if f < proba: return level
            f -= proba
        return len(self.assign_probas) - 1

    def _neighbor_range(self, node_idx, level):
        o = self.offsets[node_idx]
        begin = o + self.cum_nneighbor_per_level[level]
        end = o + self.cum_nneighbor_per_level[level + 1]
        return begin, end

    def shrink_neighbor_list(self, query_idx, input_list, max_size, keep_max_size=False):
        output = []
        outsiders = []
        for d_q_v1, v1_idx in input_list:
            good = True
            for _, v2_idx in output:
                d_v1_v2 = self._get_dist(v1_idx, v2_idx)
                if d_v1_v2 < d_q_v1:
                    good = False
                    break
            if good:
                output.append((d_q_v1, v1_idx))
                if len(output) >= max_size:
                    return [idx for d, idx in output]
            elif keep_max_size:
                outsiders.append((d_q_v1, v1_idx))
        if keep_max_size:
            for _, v1_idx in outsiders:
                if len(output) < max_size:
                    output.append((0, v1_idx))
                else: break
        return [idx for d, idx in output]

    def add_link(self, src, dest, level):
        begin, end = self._neighbor_range(src, level)
        for i in range(begin, end):
            if self.neighbors[i] == dest: return 
            if self.neighbors[i] == -1:
                self.neighbors[i] = dest
                return
        candidates = []
        for i in range(begin, end):
            v_idx = self.neighbors[i]
            candidates.append((self._get_dist(src, v_idx), v_idx))
        candidates.append((self._get_dist(src, dest), dest))
        candidates.sort()
        max_size = end - begin
        new_list = self.shrink_neighbor_list(src, candidates, max_size, keep_max_size=(level == 0))
        for i, idx in enumerate(new_list):
            self.neighbors[begin + i] = idx
        for i in range(len(new_list), max_size):
            self.neighbors[begin + i] = -1

    def add_point(self, vec, visualize_step=False):
        pt_id = len(self.data)
        self.data.append(vec)
        pt_level = self._random_level()
        self.levels.append(pt_level + 1)
        
        self.offsets.append(self.offsets[-1] + self.cum_nneighbor_per_level[pt_level + 1])
        new_slots = self.offsets[-1] - self.offsets[-2]
        self.neighbors.extend([-1] * new_slots)

        if self.entry_point == -1:
            self.entry_point = pt_id
            self.max_level = pt_level
            return

        nearest = self.entry_point
        d_nearest = self._get_dist(nearest, vec=vec)

        for level in range(self.max_level, pt_level, -1):
            changed = True
            while changed:
                changed = False
                begin, end = self._neighbor_range(nearest, level)
                for i in range(begin, end):
                    v = self.neighbors[i]
                    if v < 0: break
                    d = self._get_dist(v, vec=vec)
                    if d < d_nearest:
                        d_nearest = d
                        nearest = v
                        changed = True

        for level in range(min(pt_level, self.max_level), -1, -1):
            visited = {pt_id, nearest}
            results = [(-d_nearest, nearest)]
            candidates = [(d_nearest, nearest)]
            while candidates:
                d_curr, v_curr = heapq.heappop(candidates)
                if d_curr > -results[0][0]: break
                begin, end = self._neighbor_range(v_curr, level)
                for i in range(begin, end):
                    v_neigh = self.neighbors[i]
                    if v_neigh < 0: break
                    if v_neigh not in visited:
                        visited.add(v_neigh)
                        d_neigh = self._get_dist(v_neigh, vec=vec)
                        if len(results) < self.efConstruction or d_neigh < -results[0][0]:
                            heapq.heappush(results, (-d_neigh, v_neigh))
                            heapq.heappush(candidates, (d_neigh, v_neigh))
                            if len(results) > self.efConstruction:
                                heapq.heappop(results)
            
            candidate_list = sorted([(-d, idx) for d, idx in results])
            M_level = self.cum_nneighbor_per_level[level+1] - self.cum_nneighbor_per_level[level]
            selected = self.shrink_neighbor_list(pt_id, candidate_list, M_level, keep_max_size=(level==0))
            for neigh_id in selected:
                self.add_link(pt_id, neigh_id, level)
                self.add_link(neigh_id, pt_id, level)

        if pt_level > self.max_level:
            self.max_level = pt_level
            self.entry_point = pt_id
        
        if visualize_step:
            self.visualize(f"After adding point {pt_id}")

    def visualize(self, title_suffix=""):
        num_layers = self.max_level + 1
        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5), squeeze=False)
        fig.suptitle(f"HNSW Multi-Layer Visualization {title_suffix}")

        for level in range(num_layers):
            ax = axes[0, level]
            ax.set_title(f"Level {level}")
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            
            # Identify points that exist at this level
            points_at_level = [i for i, l in enumerate(self.levels) if l > level]
            
            # Plot points
            coords = np.array([self.data[i] for i in points_at_level])
            if len(coords) > 0:
                ax.scatter(coords[:, 0], coords[:, 1], c='blue', zorder=3)
                for i in points_at_level:
                    ax.text(self.data[i][0], self.data[i][1], str(i), fontsize=9)

            # Plot edges
            for i in points_at_level:
                begin, end = self._neighbor_range(i, level)
                for n_idx in range(begin, end):
                    neighbor = self.neighbors[n_idx]
                    if neighbor != -1:
                        p1 = self.data[i]
                        p2 = self.data[neighbor]
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', alpha=0.5, zorder=2)

        plt.tight_layout()
        plt.show(block=False)
        input("Press Enter to continue to the next point...")
        plt.close()

# ==========================================
# 3. TEST EXECUTION WITH VISUALIZATION
# ==========================================

if __name__ == "__main__":
    dim = 2
    num_elements = 10
    M = 4 # Smaller M for cleaner visualization
    
    np.random.seed(42)
    data = np.random.random((num_elements, dim)).astype(np.float32)
    
    index = FaissHNSW(dim, M)
    
    print("Starting HNSW construction with visualization...")
    for i, vec in enumerate(data):
        print(f"Adding point {i}...")
        index.add_point(vec, visualize_step=True)
        
    print("Build complete.")