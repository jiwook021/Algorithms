
#include <iostream>
#include <vector>
#include <queue>
void dfs(const std::vector<std::vector<int>>& graph, bool visited[],  int current)
{
  visited[current] = true; 
  std::cout<< current;
  for(auto now: graph[current])
  {
    if(visited[now]==false)
      dfs(graph,visited, now);
  }
}
void bfs(const std::vector<std::vector<int>>& graph, bool visited[],  int start)
{
    std::queue<int> q; 
    q.push(start); 
    visited[start] = true;
    while(!q.empty())
    {
      int current = q.front();
      std::cout << current; 
      q.pop(); 
      for(auto now: graph[q.front()])
      {  
        if(visited[now] == false)
        { 
          visited[now] = true; 
          q.push(now);
        }
      } 
    }
}
int main()
{
  int vertex, edge, start; 
  std::cin >> vertex >> edge >> start; 
  std::vector<std::vector<int>> graph(1000); 
  for(int i=0; i< edge; i++)
  {
    int src, dest;
    std::cin >> src >> dest;
    graph[src].push_back(dest); 
    graph[dest].push_back(src); 
  }
  bool visited[100];
  for(int i=1;i<=vertex;i++)
    visited[i] = false;
  //dfs(graph, visited, start);
  bfs(graph, visited, start);
  
}