from math import sqrt
import random
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

root_dir = "k-server_instances"
image_folder = "plots/"
fileSet = set()

# Define the size of the grid
grid_size = 100


class Instance:
    '''
    Instance of the k-servers problem. Should be treated as immutable.
    The given file is parsed upon construction.
    [filename]: original file
    [opt]: cost of optimal solution
    [k]: number of servers
    [sites]: positions of the sites
    [requests]: list of site ID (i.e. index of the site in [sites])
    '''

    def __init__(self, filename):
        self.filename = filename
        self.opt = 0
        self.k = 0
        self.sites = []
        self.requests = []

        with open(filename, mode='r', encoding="utf8") as file_p:

            # Check if we have parsed all sites (and then go to demands)
            ended_sites = -1

            for i, line in enumerate(file_p):

                # Get opt at line 2
                if i == 1:
                    self.opt = int(line)

                # Get k at line 5
                elif i == 4:
                    self.k = int(line)

                # Get sites from line 8 to line k (variable)
                elif i >= 7 and ended_sites == -1:
                    if line == "\n":  # No more sites
                        ended_sites = i + 2
                    else:
                        self.sites.append(list(map(int, line.strip().split(" "))))

                # Get requests from line k+2 to end of file
                # (should only be one line usually)
                elif i >= ended_sites > -1:
                    self.requests = self.requests + list(map(int, line.strip().split(" ")))
            file_p.close()


class Grid:
    '''
    Auxiliary class to solve an instance using an online algorithm.
    The same instance can be shared by multiple grids.
    [inst]: instance being solved
    [size]: size of the grid (length = width)
    [grid]: actual grid data structure
    [servers]: current position of the servers
    '''
    def __init__(self, size, inst):
        self.inst = inst
        self.size = size
        self.grid = np.zeros((size, size))
        self.servers = [(0, 0) for _ in range(inst.k)]


def make_videos_from_images(inst, algo_p):

    name = inst.filename.split("/")[1]
    folder = inst.filename.split("/")[0]
    print("Starting video : ", inst.filename, algo_p)

    images = [img for img in os.listdir(image_folder + folder) if img.endswith(".png")
                    and img.startswith(name + algo_p)]

    video_name = 'video' + name + algo_p + '.avi'
    frame = cv2.imread(image_folder  + folder + "/" + images[0])
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(video_name, fourcc, 2, (width,height))

    for image in images:
        frame = cv2.imread(image_folder  + folder + "/" + image)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()
    print("Ended video : ", video_name)


def plot_grid_and_servers(inst, nb, grid, algo, dist):
    # Create a new plot
    plt.clf()

    # Plot the grid using imshow
    plt.imshow(grid.grid, cmap='binary', vmin=0, vmax=1)

    # Plot the servers using scatter
    x, y = zip(*grid.servers)
    plt.scatter(x, y, c='r')

    # Set the plot axis limits
    plt.xlim([0, grid.grid.shape[0]])
    plt.ylim([0, grid.grid.shape[1]])
    plt.title("Algorithme : " + algo + ", distance totale : " + str(dist))
    # Show the plot
    zero_numb = 10 - len(str(nb))
    plt.savefig("plots/" + inst.filename + algo + "0"*zero_numb + str(nb) + ".png")


def distance(point1, point2):
    '''Using Manhattan distance, i.e. agents can't travel diagonally'''
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


def distances_to(grid, point):
    '''Returns a list where the [k]th entry is the distance from the [k]th agent to [point]'''
    return [distance(point, grid.servers[i]) for i in range(grid.inst.k)]


def closest_to(grid, point):
    '''Returns the agent closest to [point]'''
    dist = distances_to(grid, point)
    return dist.index(min(dist))


def run(inst, strategy, make_video = True):
    '''
    Solves [inst] using [strategy].
    '''
    total_dist = 0
    grid = Grid(grid_size, inst)

    for i, req in enumerate(inst.requests):
        req_pos = inst.sites[req]
        if make_video:
            plot_grid_and_servers(inst, i, grid, strategy.name, total_dist)
        agent = strategy.choose_agent(inst, grid, req)
        total_dist += distance(req_pos, grid.servers[agent])
        grid.servers[agent] = req_pos

    if make_video:
        make_videos_from_images(inst, strategy.name)
    return total_dist


class OneAgent:
    '''Only moves the first agent'''
    def __init__(self):
        self.name = "one"

    def choose_agent(self, inst, grid, request):
        return 0


class NearestAgent:
    '''Moves the nearest agent'''
    def __init__(self):
        self.name = "nearest"

    def choose_agent(self, inst, grid, request):
        return closest_to(grid, inst.sites[request])


class NearestAgentBis:
    '''
    Moves the nearest agent, but first each agent is positioned at the sites of the k first requests,
    so that there will be a better cover of the grid by the agents
    '''
    def __init__(self):
        self.name = "nearbis"

    def choose_agent(self, inst, grid, request):
        req_pos = inst.sites[request]
        if req_pos not in grid.servers and (0, 0) in grid.servers:
            return grid.servers.index((0, 0))
        else:
            return closest_to(grid, req_pos)


class NearestAgentTer:
    '''
    Similar to [NearestAgentBis], but if the site is too far away from the start
    (at least [limit] steps), another agent is dispatched.
    This can be useful when it is the last request in short sequences.
    '''
    def __init__(self, limit):
        self.name = "nearter"
        self.limit = limit

    def choose_agent(self, inst, grid, request):
        req_pos = inst.sites[request]
        if req_pos in grid.servers:
            return grid.servers.index(req_pos)
        else:
            dist = distances_to(grid, req_pos)
            if (0, 0) in grid.servers and distance(req_pos, (0, 0)) < self.limit:
                return grid.servers.index((0, 0))
            else:
                return dist.index(min(dist))


class PositionedThenRandom:
    '''
    Moves a random agent, but first each agent is positioned at the sites of the k first requests.
    '''
    def __init__(self):
        self.name = "posrandom"

    def choose_agent(self, inst, grid, request):
        req_pos = inst.sites[request]
        if req_pos in grid.servers:
            return grid.servers.index(req_pos)
        elif (0, 0) in grid.servers:
            return grid.servers.index((0, 0))
        else:
            return random.randint(0, inst.k-1)


class Random:
    '''Moves a random agent'''
    def __init__(self):
        self.name = "random"
    
    def choose_agent(self, inst, grid, request):
        return random.randint(0, inst.k-1)


class RandomBis:
    '''Moves a random agent, but does not move if the request is already satisfied'''
    def __init__(self):
        self.name = "randombis"
    
    def choose_agent(self, inst, grid, request):
        req_pos = inst.sites[request]
        if req_pos in grid.servers:
            return grid.servers.index(req_pos)
        else:
            return random.randint(0, inst.k-1)


class RoundRobin:
    ''' Each agent moves in turn '''
    def __init__(self):
        self.name = "roundrobin"
        self.last_moved = 0

    def choose_agent(self, inst, grid, request):
        req_pos = inst.sites[request]
        if req_pos in grid.servers:
            return grid.servers.index(req_pos)
        else:
            agent = (self.last_moved+1) % inst.k
            self.last_moved = agent
            return agent


class OwnArea:
    '''
    Each agent watch a zone (step lines)

    If we have a 4x4 grid and 4 agents, the distribution will look like that
    |1___|
    |2___|
    |3___|
    |4___|
    '''
    def __init__(self):
        self.name = "own"

    def choose_agent(self, inst, grid, request):
        req_pos = inst.sites[request]
        if req_pos in grid.servers:
            return grid.servers.index(req_pos)
        else:
            step = grid.size // inst.k
            return req_pos[0] // step


def get_squarish(number):
    ''' Get the squarish decomposition of the number
    Input n :: int
    Return (u, v) :: int, int, s.t. u*v == n and u is the closest possible
                            to the square root of n
    Note : There is always a result since n*1 == n
    '''
    squarish = int(sqrt(number))
    while (squarish*(number//squarish) != number):
        squarish += 1
    return (squarish, number//squarish)


class OwnAreaS:
    '''
    Each agent watch a zone (squarish zone)

    If we have a 4x4 grid and 4 agents, the distribution will look like that
    |1 | 2|
    |__|__|
    |  |  |
    |3 | 4|
    '''
    def __init__(self):
        self.name = "owns"

    def choose_agent(self, inst, grid, request):
        req_pos = inst.sites[request]
        if req_pos in grid.servers:
            return grid.servers.index(req_pos)
        else:
            (lines, columns) = get_squarish(inst.k)  # Get squarish decomposition of k
            l_step = grid_size // lines
            c_step = grid_size // columns
            return req_pos[0] // l_step + req_pos[1] // c_step


class PopularPlaces:
    '''
    No description (the old one was the same as nearest_agent_positioned)
    Also the [agents_spot.index(...)] seems really weird to me, because
    there are very likely more sites than servers (otherwise the problem is trivial),
    so quickly this strategy will just return the agent closest to the top left corner...
    That does not seem very interesting
    '''
    def __init__(self):
        self.name = "pop"
        self.spots = {}

    def choose_agent(self, inst, grid, request):
        req_pos = inst.sites[request]
        if request not in self.spots:
            self.spots[request] = 1
        else:
            self.spots[request] += 1
        
        if req_pos in grid.servers:
            return grid.servers.index(req_pos)
        else:
            agents_spot = [0 for _ in range(inst.k)]
            for agent in range(inst.k):
                if grid.servers[agent] in inst.sites:
                    agents_spot[agent] = self.spots[inst.sites.index(grid.servers[agent])]
            return agents_spot.index(min(agents_spot))


class PopularPlacesBis:
    '''No description'''
    def __init__(self):
        self.name = "pop2"
        self.spots = {}

    def choose_agent(self, inst, grid, request):
        req_pos = inst.sites[request]
        if request not in self.spots:
            self.spots[request] = 1
        else:
            self.spots[request] += 1

        if req_pos in grid.servers:
            return grid.servers.index(req_pos)
        else:
            agents_spot = [0 for _ in range(inst.k)]
            for agent in range(inst.k):
                if grid.servers[agent] in inst.sites:
                    agents_spot[agent] = self.spots[inst.sites.index(grid.servers[agent])]
            mini = min(agents_spot)
            less_visited = [i for i in range(inst.k) if agents_spot[i] == mini]
            dist = [(i, distance(req_pos, grid.servers[i])) for i in less_visited]
            v = min(dist, key=lambda x: x[1])
            return dist[dist.index(v)][0]


if __name__ == '__main__':

    strategies = [
        OneAgent,
        NearestAgent,
        NearestAgentBis,
        #NearestAgentTer,
        Random,
        RandomBis,
        RoundRobin,
        OwnArea,
        OwnAreaS,
        PopularPlacesBis
    ]
    results = {S.__name__: [] for S in strategies}

    # Get all test instances names
    for root, dirs, files in os.walk(root_dir):
        for fileName in files:
            fileSet.add(root_dir + "/" + fileName)

    for i, file in enumerate(fileSet):

        # Uncomment if you want to test on only one instance
        # if i != 0:
        #     break

        # Parsing file
        inst = Instance(file)

        # Header
        print("==============")
        print("Results for file ", file)
        print("Optimal value :", inst.opt)

        for S in strategies:
            score = run(inst, S(), False)
            print(f"With {S.__name__}: {score}")
            results[S.__name__].append(inst.opt*100 / score)

    print("")

    # Ordering results
    percentages = [(algo, int(np.mean(values))) for (algo, values) in results.items()]
    percentages.sort(key=lambda x: x[1])

    # Print results
    for algo, res in percentages:
        print(algo, "has a mean performance of", res, "%")
