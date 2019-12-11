import time
import numpy as np
from scipy.spatial.distance import squareform, pdist, euclidean


class dynalocTSP:
    def __init__(self, data):
        """
        Note that the self.ini_data should include the replica of the first node as the last node as the traveller has go back
        to the first city.
        :param data:
        """
        # self.ini_data = np.vstack([data, data[0]])
        # self.final_data = np.vstack([data, data[0]])
        self.ini_data = data
        self.final_data = data
        self.ini_dm = squareform(pdist(self.ini_data)).astype(int)
        self.final_dm = squareform(pdist(self.final_data)).astype(int)
        self.fullnode = list(range(len(self.ini_data)))
        self.visited_node = list()
        self.visited_node.append(0)
        self.size = len(self.ini_data)
        self.cur_node = 0
        self.findtour()
        self.visited_node.append(0)
        self.calcost()

    def findtour(self):
        while len(self.visited_node) < self.size:
            next_node = self.findNN(self.cur_node)
            # print('The next_node is', next_node)
            self.visited_node.append(next_node)
            # print('We put node', next_node, 'into the route')
            self.cur_node = next_node
            self.updatelocation()
        # else:
        #     print("We have visited all nodes")


    def findNN(self, cur_node):
        self.final_dm = squareform(pdist(self.final_data)).astype(int)
        new_dm = np.delete(self.final_dm, self.visited_node, axis=1)
        remaining_node_index = self.fullnode.copy()
        for node in self.visited_node:
            remaining_node_index.remove(node)
        next_node_index = np.argmin(new_dm[cur_node]).astype(int)
        # print(next_node_index)
        next_node = remaining_node_index[next_node_index]

        return next_node


    def updatelocation(self):
        for i in range(self.size):
            if i not in self.visited_node:
                self.final_data[i][0] += np.random.normal(0, 5)
                self.final_data[i][1] += np.random.normal(0, 5)


    def calcost(self):
        self.final_dm = squareform(pdist(self.final_data)).astype(int)
        self.total_cost = 0
        for i in range(self.size):
            cur = self.visited_node[i]
            next = self.visited_node[i+1]
            self.total_cost += self.final_dm[cur][next]



if __name__ == "__main__":
    test_sample_num = 100
    for num_nodes in [10, 20, 50, 100]:
        exec_time = 0
        cost_set = np.zeros(test_sample_num)
        print("Num nodes:", num_nodes)
        for i in range(test_sample_num):
            siz = (num_nodes, 2)
            data = np.random.randint(0, 100, size=siz)
            before = time.time()
            test = dynalocTSP(data)
            after = time.time()
            exec_time += after - before
            # print(test.visited_node)
            # print('The final cost is:', test.total_cost)
            cost_set[i] = test.total_cost

        print('Execution time for', test_sample_num, 'iterations is:', exec_time)
        print('The average cost is:', np.mean(cost_set)/100)


