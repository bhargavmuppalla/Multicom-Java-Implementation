import java.util.Comparator;

public class NodeSort implements Comparator<Node>{

    @Override
    public int compare(Node o1, Node o2) {
        //System.out.println(o2.score - o1.score);
        return (int)(o2.score - o1.score);
    }
}
