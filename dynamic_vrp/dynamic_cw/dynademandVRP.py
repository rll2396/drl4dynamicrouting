import time
import numpy as np
from scipy.spatial.distance import squareform, pdist, euclidean
from ClarkWright import ClarkWright

class dynademandVRP:
    def __init__(self, coordinate, ini_demand_set, capacity):
        self.dataprocess(coordinate, ini_demand_set, capacity)
        self.findtour()

    def dataprocess(self, coordinate, ini_demand_set, capacity):
        self.ini_demand_set = ini_demand_set.copy()
        self.final_demand_set = ini_demand_set.copy()
        self.data = coordinate
        self.distanceMatrix = squareform(pdist(self.data)).astype(int)
        self.capacity = capacity
        self.fullnodes = list(range(1,len(coordinate)))
        self.cur_loc = 0
        self.nodes_in_routes = list()
        self.final_routes = list()

    def findtour(self):
        while len(self.nodes_in_routes) != len(self.fullnodes):
            # print(self.nodes_in_routes)
            # print(self.fullnodes)
            if self.cur_loc == 0:
                newtour = list()
                newtour.append(0)
                demand = list(self.final_demand_set)
                neworder = list(range(len(self.data)))
                # print('self.nodes_in_routes is', self.nodes_in_routes)
                # print(data)
                for i in self.nodes_in_routes:
                    # data.remove(self.data[i])
                    try:
                        demand.remove(self.final_demand_set[i])
                    except:
                        print(demand)
                        print(self.final_demand_set[i])
                        assert(False)
                    neworder.remove(i)
                data = np.delete(self.data, self.nodes_in_routes, axis=0)
                # print('Now the data is', data)
                if len(data)>=3:
                    tours = ClarkWright(data, demand, self.capacity)
                    # print('the tours are:', tours)
                    chosentour = tours.routes[0]
                    # print('The chosentour is:', chosentour)
                    # the actual tour is different since the index in "tours" is different
                    actual_tour = list()
                    for item in chosentour:
                        actual_tour.append(neworder[item])
                    # print('The actual_tour is', actual_tour)
                    current_tour = list()
                    current_tour.append(0)
                    i = 1
                    while 1 <= i < len(actual_tour)-1:
                        node = actual_tour[i]
                        newtour = current_tour.copy()
                        newtour.append(node)
                        if self.calload(newtour) <= self.capacity:
                            current_tour.append(node)
                            # print('The current_tour is:', current_tour)
                            self.nodes_in_routes.append(node)
                            # print('we add node:', node, 'into the visited node set')
                            self.updatedemand()
                            self.cur_loc = actual_tour[i]
                            i += 1
                        else:
                            current_tour.append(0)
                            self.updatedemand()
                            self.cur_loc = 0
                            self.final_routes.append(current_tour)
                            # print('The current_tour is:', current_tour)
                            break

                    else:
                        current_tour.append(0)
                        # print('add 0 into the current_tour')
                        # print('The current_tour is:', current_tour)
                        self.updatedemand()
                        self.cur_loc = 0
                        self.final_routes.append(current_tour)
                else:
                    # print('The length of data is too short!')
                    # print(data)
                    for i in range(1, len(self.data)):
                        if self.data[i][0] == data[1][0] and self.data[i][1] == data[1][1]:
                            self.final_routes.append([0, i, 0])
                            self.nodes_in_routes.append(i)

        # else:
        #     print('We have successfully put all the nodes in the route')
            # print(self.final_routes)


    def calload(self, tour):
        load = 0
        for i in range(1, len(tour)):
            load += self.final_demand_set[i]
        return load


    def updatedemand(self):
        """
        This function aims to update the demand of each node every time when a node is put in a route
        :return:
        """
        updateindex = [item for item in self.fullnodes if item not in self.nodes_in_routes]
        for index in updateindex:
            num = np.random.random()
            if 0 < num <= 0.3333:
                self.final_demand_set[index] -= 1
            elif 0.6667 < num <= 1:
                self.final_demand_set[index] += 1

        for i in range(len(self.final_demand_set)):
            cur = self.final_demand_set[i]
            if cur < 1:
                self.final_demand_set[i] = 1
            if cur > 9:
                self.final_demand_set[i] = 9




if __name__ == "__main__":
    num_trials = 100
    for num_nodes in [10, 20, 50, 100]:
        cost_set = []
        print("Num nodes:", num_nodes)
        exec_time = 0
        for _ in range(num_trials):
            demand_set = np.random.randint(1, 10, num_nodes)
            demand_set[0] = 0
            siz = (num_nodes, 2)
            data = np.random.randint(0, 100, size=siz)
            capacity = 30

            before = time.time()
            test = dynademandVRP(data, demand_set, capacity)
            after = time.time()
            exec_time += after - before
            # print('The final routes is:',test.final_routes)
            dm = test.distanceMatrix
            total_cost = 0
            for tour in test.final_routes:
                for i in range(len(tour)-1):
                    total_cost += dm[tour[i]][tour[i+1]]

            cost_set.append(total_cost)

        cost_array = np.array(cost_set)
        print('Execution time for', num_trials, 'trials:', exec_time)
        print('The average cost is:', np.mean(cost_array)/100)



