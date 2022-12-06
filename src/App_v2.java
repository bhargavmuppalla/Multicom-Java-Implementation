import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Iterator;

import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;
import org.apache.commons.math3.ml.clustering.DoublePoint;

public class App_v2 {
    static class Pair{
        public int u,v;
        public Pair(int u, int v)
        {
            // This keyword refers to current instance
            this.u = u;
            this.v = v;
        }
    }

    static void transpose(int A[][], int AT[][])
    {
        int i, j;
        for (i = 0; i < A.length; i++)
            for (j = 0; j < A[0].length; j++)
                AT[i][j] = A[j][i];
    }

    public static void add_edge(Map<Integer, ArrayList<Integer>> adj_list, int u, int v){
        if(adj_list.containsKey(u))
        {
            ArrayList<Integer> temp = adj_list.get(u);
            temp.add(v);
            adj_list.put(u, temp);
        }
        else
        {
            ArrayList<Integer> temp = new ArrayList<>();
            temp.add(v);
            adj_list.put(u, temp);
        }
    }

    /**
     * reads edgelist and returns adjacency list representation of the graph.
     * @param edgelist_filename
     * @param delimiter
     * @return
     * @throws FileNotFoundException
     */
    public static Map<Integer, ArrayList<Integer>> load_graph(String edgelist_filename, String delimiter) throws FileNotFoundException{
        
        // ArrayList<Pair> edge_list = new ArrayList<>();
        
        BufferedReader br = new BufferedReader(new FileReader(edgelist_filename)); 
        String line = "";
        // int maxNode = -1;
        Map<Integer, ArrayList<Integer>> adj_list = new HashMap<>();
        try {
            while ((line = br.readLine()) != null) 
            {  
                String[] edge = line.split(delimiter);   
                int u = Integer.parseInt(edge[0]);
                int v = Integer.parseInt(edge[1]); 

                add_edge(adj_list, u, v);
                add_edge(adj_list, v, u);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } 
        return adj_list;
    }

   
    /**
     * Load a collection of ground-truth communities.
     * @param edgelist_filename
     * @param delimiter
     * @return
     * @throws FileNotFoundException
     */
    public static ArrayList<HashSet<Integer>> load_groundtruth(String edgelist_filename, String delimiter) throws FileNotFoundException{
        ArrayList<HashSet<Integer>> groundTruthCommunities = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(edgelist_filename)); 
        String line = "";
        try {
            while ((line = br.readLine()) != null) 
            {  
                // System.out.println(line);
                String[] sArray = line.split(delimiter);
                HashSet<Integer> community = new HashSet<>();
                for(String s: sArray){
                    community.add(Integer.parseInt(s));
                }
                groundTruthCommunities.add(community);
            }
            // System.out.println(groundTruthCommunities);
        } catch (IOException e) {
            e.printStackTrace();
        } 
        return groundTruthCommunities;
    }

    /**
     * Get the community membership for each node given a list of communities.
     * @param communities
     * @return
     */
    public static Map<Integer, Integer> get_node_membership(ArrayList<HashSet<Integer>> communities)
    {
        Map<Integer, Integer> membership = new HashMap<>();
        for(int i = 0; i < communities.size(); i++)
        {
            HashSet<Integer> community = communities.get(i);
            for(Integer node: community){
                membership.put(node, i);
            }
        }
        return membership;
    }
    /**
     * Compute the maximum F1-Score for each community of a list of communities
     * with respect to a collection of ground-truth communities.
     * @param communities
     * @param groundTruthCommunities
     * @return 
     */
    public static List<Double> compute_f1_scores(ArrayList<HashSet<Integer>> communities, ArrayList<HashSet<Integer>> groundTruthCommunities)
    {
        Map<Integer,Integer> groundtruth_inv = get_node_membership(groundTruthCommunities);
        List<Double> f1_scores = new ArrayList<>();
        for(HashSet<Integer> community: communities){
            
            HashSet<Integer> groundtruth_indices  = new HashSet<>();
            for(Integer c: community){
                if(groundtruth_inv.get(c) != null)
                    groundtruth_indices.add(groundtruth_inv.get(c));
            }

            double max_precision = 0.0, max_recall = 0.0, max_f1 = 0.0;
            
            for(Integer i: groundtruth_indices){
                double count = 0.0;
                for(Integer j: groundTruthCommunities.get(i)){
                    if(community.contains(j))
                        count += 1.0;
                }
                double precision = count/(double)community.size();
                double recall = count/(double)groundTruthCommunities.get(i).size();
                double f1 = 2 * precision * recall / (precision + recall);
                max_precision = Math.max(precision, max_precision);
                max_recall = Math.max(recall, max_recall);
                max_f1 = Math.max(f1, max_f1);
                f1_scores.add(max_f1);
            }
        }
        return f1_scores;
    }

    /**
     * Compute the approximate Personalized PageRank (PPR) from a set of seed node.
     * This function implements the push method introduced by Andersen et al.
     * in "Local graph partitioning using pagerank vectors", FOCS 2006.
     * @param adj_matrix
     * @param seedset
     * @param alpha = 0.85
     * @param epsilon = 1e-5
     * @return
     */
    public static Map<Integer, Double> approximate_ppr(Map<Integer, ArrayList<Integer>> adj_matrix, ArrayList<Integer> seed_set ,double alpha, double epsilon){
        Map<Integer,Integer> degree = getDegrees(adj_matrix);
 
        Map<Integer,Double> prob = new HashMap<>();
        Map<Integer,Double> res = new HashMap<>();

        for(Integer seed: seed_set){
            double seedV = 1.0/seed_set.size();
            res.put(seed, seedV);
        }
        Deque<Integer> next_nodes = new LinkedList<>(seed_set);

        while(next_nodes.size() > 0){
            int node = next_nodes.pop();
            int node_degree = 0;
            if(degree.containsKey(node))
                node_degree = degree.get(node);
            double push_val = res.get(node) - 0.5 * epsilon * node_degree;
            double resV = 0.5 * epsilon * node_degree;
            res.put(node, resV);
            double probV = (1.0 - alpha) * push_val;
            prob.put(node, prob.getOrDefault(node, 0.0)+probV); 
            double put_val = alpha * push_val;
            ArrayList<Integer> neighbours = getNeighbours(adj_matrix, node);
            if(neighbours == null)
                continue;
            for(Integer neighbour : neighbours){
                double old_res = 0.0;
                if(res.containsKey(neighbour))
                    old_res = res.get(neighbour);

                int neighborCount = 0;
                if(adj_matrix.get(node).contains(neighbour))
                    neighborCount = 1;

                double resNeighbourV = put_val * neighborCount / degree.get(node);
                res.put(neighbour, res.getOrDefault(neighbour, 0.0)+resNeighbourV);
                double threshold = epsilon * degree.get(neighbour);
                if(res.get(neighbour) >= threshold && threshold > old_res)
                    next_nodes.addFirst(neighbour);
            }
        }
        return prob;
    }

    /**
     * returns neighbors of node
     * @param adj_matrix
     * @param node
     * @return
     */
    private static ArrayList<Integer> getNeighbours(Map<Integer, ArrayList<Integer>> adj_matrix, int node) {
        return adj_matrix.get(node);
    }

    /**
     * return degree of node
     * @param adj_matrix
     * @return
     */
    private static Map<Integer,Integer> getDegrees(Map<Integer, ArrayList<Integer>> adj_matrix) {
        Map<Integer,Integer> degree = new HashMap<>();

        for(Map.Entry<Integer, ArrayList<Integer>> e: adj_matrix.entrySet()){
            degree.put(e.getKey(),e.getValue().size());
        }
        return degree;
    }

    /**
     * returns multiple local communities
     * @param adj_matrix
     * @param n_steps
     */
    public static ArrayList<HashSet<Integer>> multicom(Map<Integer, ArrayList<Integer>> adj_matrix,int n_steps, ArrayList<Integer> seed_set, double explored_ratio){
        n_steps = 5;
        int step = -1;
        int n_iter = 0;
        int n_nodes = adj_matrix.size();

        ArrayList<Integer> new_seeds  = new ArrayList<>(seed_set);
        Map<Integer,ArrayList<Integer>> seeds = new HashMap<>();
        Map<Integer,Map<Integer, Double>> scores = new HashMap<>();
        Map<Double, Integer> scoresMap = new HashMap<>();
        ArrayList<Integer> explored = new ArrayList<>();
        ArrayList<HashSet<Integer>> communities = new ArrayList<>();

        while (step < n_steps && new_seeds.size() > 0){
            n_iter += 1;

            for(Integer new_seed : new_seeds){
                // System.out.println(new_seed);
                step += 1;
                ArrayList<Integer> seedList = new ArrayList<>();
                seedList.add(new_seed);
                seeds.put(step,seedList);
                Map<Integer, Double> pprscores = approximate_ppr(adj_matrix,seeds.get(step), 0.85, 0.00001);
                for(Map.Entry<Integer,Double> e: pprscores.entrySet()){
                    scoresMap.put(e.getValue(), e.getKey());
                }
                scores.put(step,pprscores);
                HashSet<Integer> community = conductance_sweep_cut(adj_matrix, scores.get(step), 10);
                communities.add(community);

                for(Integer i: community){
                    explored.add(i);
                }
            }
            new_seeds.clear();
            ArrayList<DoublePoint> embedding = new ArrayList<>();
            for(int j = 0; j <  scores.get(0).size(); j++){
                double[] embeddingScores = new double[scores.size()];
                for(int k = 0; k < scores.size(); k++)
                {
                    double score = 0;
                    Map<Integer, Double> smap = scores.get(k);
                    if(smap != null && smap.containsKey(j))
                        score = scores.get(k).get(j);

                    embeddingScores[k] = score;
                }
                embedding.add(new DoublePoint(embeddingScores));
            }
            DBSCANClusterer<DoublePoint> dbscan = new DBSCANClusterer<>(0.5, 5);
            List<Cluster<DoublePoint>> y = dbscan.cluster(embedding);
            Map<Integer, ArrayList<Integer>> clusters = new HashMap<>();

            for(int i = 0; i < y.size(); i++)
            {
                List<DoublePoint> dp = y.get(i).getPoints();
                ArrayList<Integer> points = new ArrayList<>();
                for(int j = 0; j < dp.size(); j++){
                    double cs = dp.get(j).getPoint()[0];
                    if(scoresMap.containsKey(cs))
                        points.add(scoresMap.get(cs));
                }
               clusters.put(i, points); 
            }

            for(Map.Entry<Integer,ArrayList<Integer>> c: clusters.entrySet()){
                int cluster_size = 0;
                int cluster_explored = 0;
                for(Integer node: c.getValue()){
                    cluster_size += 1;
                    if(explored.contains(node)){
                        cluster_explored += 1;
                    }
                }
                if((double)cluster_explored / (double)cluster_size < explored_ratio)
                {
                        ArrayList<Integer> candidates = c.getValue();
                        Iterator<Integer> iterator = explored.iterator();
                        while(iterator.hasNext()){
                            candidates.remove(iterator.next());
                        }
                        // System.out.println(candidates);
                        ArrayList<Integer> candidate_degrees = new ArrayList<>();
                        Map<Integer, Integer> degree = getDegrees(adj_matrix);
                        int maxDegree = Integer.MIN_VALUE;
                        int maxDegreeNode = -1;
                        for(int i = 0; i < candidates.size(); i++)
                        {
                            //Find index of node with maximum degree
                            int node = candidates.get(i);
                            int node_degree = 0;
                            if(degree.containsKey(node))
                                node_degree = degree.get(node);
                            // else
                            //     System.out.println(node);
                            int ndegree = node_degree;
                            if(maxDegree < ndegree)
                            {
                                maxDegreeNode = i;
                                maxDegree = ndegree;
                            }
                            candidate_degrees.add(ndegree);

                        }
                        int newseed = candidates.get(maxDegreeNode);
                        new_seeds.add(newseed);
                }
            }
        }
        return communities;
    }

    /**
     * Return the sweep cut for conductance based on a given score.
     * During the sweep process, we detect a local minimum of conductance using a given window.
     * The sweep process is described by Andersen et al. in
     * "Communities from seed sets", 2006.
     * @param adj_matrix
     * @param map
     * @param window
     */
    private static HashSet<Integer> conductance_sweep_cut(Map<Integer, ArrayList<Integer>> adj_matrix, Map<Integer, Double> map, int window) {

        Map<Integer,Integer> degreeMap = getDegrees(adj_matrix);
        int total_volume = getTotalValue(degreeMap);
        
        ArrayList<Node> sorted_nodes = new ArrayList<>();
        for(Map.Entry<Integer, ArrayList<Integer>> e: adj_matrix.entrySet()){
            if(map.containsKey(e.getKey()) && map.get(e.getKey()) > 0)
                sorted_nodes.add(new Node(e.getKey(), map.get(e.getKey())));
        }
        sorted_nodes.sort(new Comparator<Node>() {
            public int compare(Node a, Node b){
                if(a.score > b.score)
                    return -1;
                else if(a.score < b.score)
                    return 1;
                else
                    return -1;
            }
        });
        ArrayList<Integer> s_nodes = new ArrayList<>();
        for(Node n: sorted_nodes){
            s_nodes.add(n.node);
        }

        HashSet<Integer> sweep_set = new HashSet<>();
        double volume = 0.0;
        double cut = 0.0;
        double best_conductance = 1.0;
        
        HashSet<Integer> best_sweep_set = new HashSet<>();
        best_sweep_set.add(s_nodes.get(0));

        int inc_count = 0;

        for(Integer node: s_nodes){
            volume += degreeMap.get(node);
            ArrayList<Integer> neighbors = getNeighbours(adj_matrix, node);
            for(Integer i: neighbors){
                if(sweep_set.contains(i)){
                    cut -= 1;
                }
                else{
                    cut += 1;
                }
            }
            sweep_set.add(node);
            
            if(volume == total_volume)
            {
                break;
            }
            double conductance = cut / Math.min(volume, total_volume - volume);
            if(conductance < best_conductance){
                best_conductance = conductance;
                best_sweep_set = new HashSet<>(sweep_set);
                inc_count = 0;
            }
            else{
                if(inc_count >= window)
                    break;
            }
        }
        return best_sweep_set; 
    }

    /**
     * returns sum of degrees of all nodes.
     * @param degree
     * @return
     */
    private static int getTotalValue(Map<Integer,Integer> degree) {
        int volume = 0;
        for(Map.Entry<Integer,Integer> m : degree.entrySet())
            volume += m.getValue();
        
        return volume;
    }

    public static void main(String[] args) throws Exception {
        
        //Function to evaluate our algorithm on DBLP dataset
        Test("data/com-dblp.ungraph.txt","data/com-dblp.all.cmty.txt");

        //Function to evaluate our algorithm on Amazon dataset. uncomment next line to execute
        // Test("data/com-amazon.ungraph.txt","data/com-amazon.all.dedup.cmty.txt");

        //Function to evaluate our algorithm on YouTube dataset. uncomment next line to execute
        // Test("data/com-youtube.ungraph.txt", "data/com-youtube.all.cmty.txt");
        
}

private static void Test(String edgelist, String groundtruthfile) throws FileNotFoundException {
        Map<Integer, ArrayList<Integer>> adj_matrix = load_graph(edgelist,"\t");
        int count = 0;
        double minscore = 1.0;
        double maxscore = 0.0;
        for(Map.Entry<Integer, ArrayList<Integer>> e: adj_matrix.entrySet()){
            //change count value to 1 or 2 for youtube dataset to see the result. 
            // there are some missing values for first 30 nodes in youtube dataset.
            if(count == 30)
                break;
            ArrayList<Integer> seed_set = new ArrayList<>();
            seed_set.add(e.getKey());
            ArrayList<HashSet<Integer>> communities = multicom(adj_matrix, 5, seed_set, 0.9);
            ArrayList<HashSet<Integer>> groundtruth = load_groundtruth(groundtruthfile, "\t");
            List<Double> f1_scores = compute_f1_scores(communities, groundtruth);
           
            double f1_score_avg = 0.0;
            for(double s: f1_scores){
                f1_score_avg += s;
            }
            f1_score_avg = f1_score_avg/f1_scores.size();
            if(f1_score_avg > maxscore)
                maxscore = f1_score_avg;
            if(f1_score_avg < minscore)
                minscore = f1_score_avg;

            // System.out.println(f1_score_avg);
            count++;
        }
        double F1range = (maxscore+minscore)/2.0;
        System.out.println("F1score Midvalue for 30 random nodes "+F1range);
        System.out.println("F1score maximum range " +(maxscore - F1range));
}
}


