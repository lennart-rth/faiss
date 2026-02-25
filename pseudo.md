ALGORITHM: search_neighbors_to_add
INPUTS:
  - hnsw: The graph structure
  - qdis: Distance computer (calculates dist(query, node))
  - entry_point: The nearest node found in the layer above
  - d_entry_point: Distance from query to entry_point
  - level: The current layer being built
  - vt: VisitedTable (bitset to track visited nodes)

VARIABLES:
  - results: Max-Priority Queue (holds top efConstruction closest nodes found)
  - candidates: Min-Priority Queue (holds nodes to expand, ordered by closest)
  - buffer_ids[4]: Array to hold neighbors for batch processing

BEGIN
  1. Initialize
     Push (d_entry_point, entry_point) into candidates
     Push (d_entry_point, entry_point) into results
     Mark entry_point as visited in vt

  2. Search Loop
     WHILE candidates is not empty:

       a. Get Closest Candidate
          curr_dist, curr_node = Pop min from candidates

       b. Pruning Condition (Lower Bound Optimization)
          IF curr_dist > results.top_distance():
             BREAK loop
             (Since candidates is sorted, all remaining candidates are worse 
              than the worst node in our current 'best result' list. 
              Further expansion is futile.)

       c. Neighbor Expansion
          Get neighbor_range (begin, end) for curr_node at current level
          
          Initialize batch_counter = 0

          FOR EACH neighbor_id in neighbor_range:
             
             i. Check Visited
                IF neighbor_id is visited in vt:
                   CONTINUE
                Mark neighbor_id as visited in vt

             ii. Batching (SIMD Optimization)
                 Add neighbor_id to buffer_ids
                 Increment batch_counter

                 IF batch_counter == 4:
                    Compute distances for all 4 ids in buffer_ids using SIMD
                    FOR EACH (id, dist) in batch:
                       CALL Update_Results(id, dist)
                    Reset batch_counter = 0

          d. Process Leftovers
             IF batch_counter > 0:
                Compute distances for remaining ids
                FOR EACH (id, dist) in leftovers:
                   CALL Update_Results(id, dist)

     END WHILE

END ALGORITHM

SUBROUTINE: Update_Results(id, dist)
  IF results.size < efConstruction OR dist < results.top_distance():
     Push (dist, id) into results
     Push (dist, id) into candidates
     
     IF results.size > efConstruction:
        Pop max from results (remove worst candidate to keep size constant)



###################################################################################
ALGORITHM: search_from_candidates
INPUTS:
  - hnsw: The graph structure
  - qdis: Distance computer
  - res: ResultHandler (Manages the top-K results and dynamic search radius)
  - candidates: MinimaxHeap (The search frontier, size efSearch)
  - vt: VisitedTable
  - level: Current level (usually 0)

VARIABLES:
  - threshold: Current search radius (dynamic, managed by 'res')
  - no_improvement_counter: Tracks unproductive steps for Early Stopping

BEGIN
  1. Initial Candidate Processing
     (The candidates heap is pre-filled with entry points from upper layers)
     threshold = res.threshold
     FOR EACH node in candidates:
        IF node matches IDSelector:
           IF dist < threshold:
              res.add_result(dist, node) (This might lower the threshold)
        Mark node as visited in vt

  2. Search Loop
     WHILE candidates is not empty:

       a. Pop Nearest
          d0, v0 = Pop min from candidates

       b. Distance Check (Standard HNSW Stop Condition)
          count = number of visited nodes in candidates with dist < d0
          IF count >= efSearch:
             BREAK loop
             (We have exhausted the exploration budget 'efSearch' around the query)

       c. Neighbor Loop Setup
          Get neighbors of v0 at current level
          
          [Optimization] Prefetch L2 cache for neighbor data

       d. Batch Processing Loop (SIMD)
          Initialize batch_counter = 0
          improved_in_this_step = FALSE

          FOR EACH neighbor v1 of v0:
             
             i. Check Visited
                IF vt.get(v1) is TRUE:
                   CONTINUE
                Mark v1 as visited

             ii. Buffer for SIMD
                 buffer_ids[batch_counter] = v1
                 batch_counter++

             iii. Execute Batch (if full)
                 IF batch_counter == 4:
                    Calculate 4 distances at once (qdis.distances_batch_4)
                    FOR EACH id, dist in batch:
                       Process_Neighbor(id, dist)
                    Reset batch_counter = 0

          e. Process Leftovers
             FOR EACH remaining id, dist:
                Process_Neighbor(id, dist)

       e. Naive Early Stopping (Custom Feature in this Code)
          IF level == 0 AND use_naive_es is TRUE:
             IF improved_in_this_step is TRUE:
                no_improvement_counter = 0
             ELSE:
                no_improvement_counter++
             
             IF no_improvement_counter >= es_patience:
                BREAK loop

     END WHILE

  3. Update Stats
     Increment total distance calculations, hops, etc.

END ALGORITHM

SUBROUTINE: Process_Neighbor(id, dist)
   update threshold = res.threshold
   
   1. Update Result Handler (Top-K)
      IF id is valid AND dist < threshold:
         IF res.add_result(dist, id) is TRUE:
            improved_in_this_step = TRUE
            threshold = res.threshold (Radius shrinks)
   
   2. Update Search Frontier (Candidate Heap)
      candidates.push(id, dist)
      (The MinimaxHeap automatically maintains size 'efSearch' 
       and discards the worst candidates)