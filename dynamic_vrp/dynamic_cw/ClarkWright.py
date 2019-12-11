import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist, euclidean

class ClarkWright:
    def __init__(self,coordinate, demand_set, capacity):
        self.dataprocess(coordinate, demand_set, capacity)
        self.findtour()


    def dataprocess(self, coordinate, demand_set, capacity):
        num_nodes = len(coordinate)
        self.demand_set = demand_set
        self.data = coordinate
        self.distanceMatrix = squareform(pdist(self.data)).astype(float)
        self.capacity =capacity

        # calculate savings for each link
        savings = dict()
        for r in range(1, len(self.data)):
            for c in range(1, len(self.data)):
                if int(c) != int(r):
                    a = max(int(r), int(c))
                    b = min(int(r), int(c))
                    key = '(' + str(a) + ',' + str(b) + ')'
                    savings[key] = self.distanceMatrix[0][int(r)] + self.distanceMatrix[0][int(c)] - self.distanceMatrix[c][r]

        # put savings in a pandas dataframe, and sort by descending
        sv = pd.DataFrame.from_dict(savings, orient='index')
        sv.rename(columns={0: 'saving'}, inplace=True)
        sv.sort_values(by=['saving'], ascending=False, inplace=True)
        sv.head()
        self.sv = sv

    def findtour(self):

        def get_node(link):
            link = link[1:]
            link = link[:-1]
            nodes = link.split(',')
            return [int(nodes[0]), int(nodes[1])]

        # determine if a node is interior to a route
        def interior(node, route):
            try:
                i = route.index(node)
                # adjacent to depot, not interior
                if i == 0 or i == (len(route) - 1):
                    label = False
                else:
                    label = True
            except:
                label = False
            return label

        # merge two routes with a connection link
        def merge(route0, route1, link):
            if route0.index(link[0]) != (len(route0) - 1):
                route0.reverse()
            if route1.index(link[1]) != 0:
                route1.reverse()
            return route0 + route1

        # sum up to obtain the total passengers belonging to a route
        def sum_cap(route):
            sum_cap = 0
            for node in route:
                sum_cap += self.demand_set[node]
            return sum_cap

        # determine 4 things:
        # 1. if the link in any route in routes -> determined by if count_in > 0
        # 2. if yes, which node is in the route -> returned to node_sel
        # 3. if yes, which route is the node belongs to -> returned to route id: i_route
        # 4. are both of the nodes in the same route? -> overlap = 1, yes; otherwise, no
        def which_route(link, routes):
            # assume nodes are not in any route
            node_sel = list()
            i_route = [-1, -1]
            count_in = 0
            for route in routes:
                for node in link:
                    try:
                        route.index(node)
                        i_route[count_in] = routes.index(route)
                        node_sel.append(node)
                        count_in += 1
                    except:
                        a = 1
            if i_route[0] == i_route[1]:
                overlap = 1
            else:
                overlap = 0
            return node_sel, count_in, i_route, overlap

        # create empty routes
        routes = list()

        # if there is any remaining customer to be served
        remaining = True

        # determine capacity of the vehicle
        cap = self.capacity

        # record steps
        step = 0
        # get a list of nodes, excluding the depot
        node_list = list(range(1, len(self.data)))
        # run through each link in the saving list
        for link in self.sv.index:
            step += 1
            if remaining:
                link = get_node(link)
                node_sel, num_in, i_route, overlap = which_route(link, routes)

                # condition a. Either, neither i nor j have already been assigned to a route,
                # ...in which case a new route is initiated including both i and j.
                if num_in == 0:
                    if sum_cap(link) <= cap:
                        routes.append(link)
                        node_list.remove(link[0])
                        node_list.remove(link[1])
                elif num_in == 1:
                    n_sel = node_sel[0]
                    i_rt = i_route[0]
                    position = routes[i_rt].index(n_sel)
                    link_temp = link.copy()
                    link_temp.remove(n_sel)
                    node = link_temp[0]

                    cond1 = (not interior(n_sel, routes[i_rt]))
                    cond2 = (sum_cap(routes[i_rt] + [node]) <= cap)

                    if cond1:
                        if cond2:
                            if position == 0:
                                routes[i_rt].insert(0, node)
                            else:
                                routes[i_rt].append(node)
                            node_list.remove(node)
                else:
                    if overlap == 0:
                        cond1 = (not interior(node_sel[0], routes[i_route[0]]))
                        cond2 = (not interior(node_sel[1], routes[i_route[1]]))
                        cond3 = (sum_cap(routes[i_route[0]] + routes[i_route[1]]) <= cap)

                        if cond1 and cond2:
                            if cond3:
                                route_temp = merge(routes[i_route[0]], routes[i_route[1]], node_sel)
                                temp1 = routes[i_route[0]]
                                temp2 = routes[i_route[1]]
                                routes.remove(temp1)
                                routes.remove(temp2)
                                routes.append(route_temp)
                                if link[0] in node_list:
                                    node_list.remove(link[0])
                                if link[1] in node_list:
                                    node_list.remove(link[1])
            else:
                break
            remaining = bool(len(node_list) > 0)
        # check if any node is left, assign to a unique route
        for node_o in node_list:
            routes.append([node_o])

        for route in routes:
            route.insert(0, 0)
            route.append(0)

        self.routes = routes
