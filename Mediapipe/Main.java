import java.io.*;
import java.util.*;

public class Main{
    static class FastReader{
        BufferedReader br;
        StringTokenizer st;
        public FastReader(){
            br=new BufferedReader(new InputStreamReader(System.in));
        }
        String next(){
            while(st==null || !st.hasMoreTokens()){
                try {
                    st=new StringTokenizer(br.readLine());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            return st.nextToken();
        }
        int nextInt(){
            return Integer.parseInt(next());
        }
        long nextLong(){
            return Long.parseLong(next());
        }
        double nextDouble(){
            return Double.parseDouble(next());
        }
        String nextLine(){
            String str="";
            try {
                str=br.readLine().trim();
            } catch (Exception e) {
                e.printStackTrace();
            }
            return str;
        }
    }
    static class FastWriter {
		private final BufferedWriter bw;

		public FastWriter() {
			this.bw = new BufferedWriter(new OutputStreamWriter(System.out));
		}

		public void print(Object object) throws IOException {
			bw.append("" + object);
		}

		public void println(Object object) throws IOException {
			print(object);
			bw.append("\n");
		}

		public void close() throws IOException {
			bw.close();
		}
	}
    public static void main(String[] args) {
        try {
            FastReader sc=new FastReader();
            FastWriter out = new FastWriter();
            int t=sc.nextInt();
            for(int m=0;m<t;m++)
            {
                int n=sc.nextInt();
                int a[]=new int[n];
                for(int i=0;i<n;i++)
                {
                    a[i]=sc.nextInt();
                    a[i]=a[i]-i-1;
                }
                int max=0;
                for(int i=1;i<n;i++)
                {
                    if(a[max]<a[i])
                    {
                        max=i;
                    }
                }
                out.println(max+1);
            }
            out.close();
        } catch (Exception e) {
    		e.printStackTrace();
	  }
    }
}
