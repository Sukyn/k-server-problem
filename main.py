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

# Create the 2D grid using NumPy
grid = np.zeros((grid_size, grid_size))


def make_videos_from_images(inst_v, algo_p):

    name = inst_v.split("/")[1]
    folder = inst_v.split("/")[0]
    print("Starting video : ", inst_v, algo_p)

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

def plot_grid_and_servers(inst, nb, grid, servers, algo, dist):
    # Create a new plot
    plt.clf()

    # Plot the grid using imshow
    plt.imshow(grid, cmap='binary', vmin=0, vmax=1)

    # Plot the servers using scatter
    x, y = zip(*servers)
    plt.scatter(x, y, c='r')

    # Set the plot axis limits
    plt.xlim([0, grid.shape[0]])
    plt.ylim([0, grid.shape[1]])
    plt.title("Algorithme : " + algo + ", distance totale : " + str(dist))
    # Show the plot
    zero_numb = 10 - len(str(nb))
    plt.savefig("plots/" + inst + algo + "0"*zero_numb + str(nb) + ".png")



def parse_file(file_name):

    opt_parse = 0
    k_parse = 0
    sites_parse = []
    demands_parse = []

    with open(file_name, mode='r', encoding="utf8") as file_p:

        # Check if we have parsed all sites (and then go to demands)
        ended_sites = -1

        for i, line in enumerate(file_p):

            # Get opt at line 2
            if i == 1:
                opt_parse = int(line)

            # Get k at line 5
            elif i == 4:
                k_parse = int(line)

            # Get sites from line 8 to line k (variable)
            elif i >= 7 and ended_sites == -1:
                if line == "\n":  # No more sites
                    ended_sites = i + 2
                else:
                    sites_parse.append(list(map(int, line.strip().split(" "))))

            # Get requests from line k+2 to end of file
            # (should only be one line usually)
            elif i >= ended_sites > -1:
                demands_parse = demands_parse + list(map(int, line.strip().split(" ")))

        file_p.close()
    return opt_parse, k_parse, sites_parse, demands_parse


def distance(point1, point2):
    '''Using Manhattan distance, i.e. agents can't travel diagonals'''
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


def one_agent(inst_one, k_one, sites_one, demands_one):
    '''Only move one agent'''

    total_dist = 0
    iteration = 0
    pos = [[0, 0] for _ in range(k_one)]

    for dem in demands_one:

        req_pos = sites_one[dem]

        plot_grid_and_servers(inst_one, iteration, grid, pos, "one", total_dist)
        iteration += 1

        total_dist += distance(req_pos, pos[0])
        pos[0] = req_pos

    make_videos_from_images(inst_one, "one")
    return total_dist


def use_nearest_agent(inst_near, k_near, sites_near, demands_near):
    '''Move the nearest agent'''

    total_dist = 0
    pos = [[0, 0] for _ in range(k_near)]
    iteration = 0

    for dem in demands_near:

        req_pos = sites_near[dem]

        plot_grid_and_servers(inst_near, iteration, grid, pos, "nearest", total_dist)
        iteration += 1

        # Compute distances
        dist = [distance(req_pos, pos[i]) for i in range(k_near)]
        moving_agent = dist.index(min(dist))  # Get the nearest agent

        total_dist += dist[moving_agent]
        pos[moving_agent] = req_pos

    make_videos_from_images(inst_near, "nearest")
    return total_dist


def use_nearest_agent_positioned(inst_near_p, k_near_p, sites_near_p, demands_near_p):
    '''Move the nearest agent
    We first position each agent on the diagonal axis such that each agent
        is activated'''

    total_dist = 0
    pos = [[0, 0] for _ in range(k_near_p)]
    step = (grid_size//k_near_p)
    iteration = 0

    # Setting up agents
    for agent in range(k_near_p):

        plot_grid_and_servers(inst_near_p, iteration, grid, pos, "nearestpos", total_dist)
        iteration += 1

        agent_pos = [agent*step, agent*step]
        total_dist += distance([0, 0], agent_pos)
        pos[agent] = agent_pos

    for dem in demands_near_p:

        req_pos = sites_near_p[dem]

        plot_grid_and_servers(inst_near_p, iteration, grid, pos, "nearestpos", total_dist)
        iteration += 1

        # Compute distances
        dist = [distance(req_pos, pos[i]) for i in range(k_near_p)]
        moving_agent = dist.index(min(dist))  # Get the nearest one

        total_dist += dist[moving_agent]
        pos[moving_agent] = req_pos

    make_videos_from_images(inst_near_p, "nearestpos")
    return total_dist


def use_nearest_agent_positioned_bis(inst_near_p2, k_near_p2, sites_near_p2, demands_near_p2):
    '''Move the nearest agent
    We first position each agent at a site when the first requests arrive
        so that there will be more agents moving and sites covered at the
        same time'''

    total_dist = 0
    pos = [[0, 0] for _ in range(k_near_p2)]
    iteration = 0

    for dem in demands_near_p2:

        req_pos = sites_near_p2[dem]

        plot_grid_and_servers(inst_near_p2, iteration, grid, pos, "nearposbis", total_dist)
        iteration += 1

        if req_pos not in pos and [0, 0] in pos:

            moving_agent = pos.index([0, 0])
            total_dist += distance([0, 0], req_pos)

        else:

            dist = [distance(req_pos, pos[i]) for i in range(k_near_p2)]
            moving_agent = dist.index(min(dist))

            total_dist += dist[moving_agent]

        pos[moving_agent] = req_pos

    make_videos_from_images(inst_near_p2, "nearposbis")
    return total_dist


def positioned_then_random(inst_random_p, k_random_p, sites_random_p, demands_random_p):
    '''Move a random agent
    We first position each agent at a site when the first requests arrive
        so that there will be more agents moving and sites covered at the
        same time'''

    total_dist = 0
    pos = [[0, 0] for _ in range(k_random_p)]
    iteration = 0

    for dem in demands_random_p:

        req_pos = sites_random_p[dem]
        plot_grid_and_servers(inst_random_p, iteration, grid, pos, "posrandom", total_dist)
        iteration += 1

        if req_pos not in pos:
            if [0, 0] in pos:
                moving_agent = pos.index([0, 0])
                total_dist += distance([0, 0], req_pos)
            else:
                moving_agent = random.randint(0, k_random_p-1)
                total_dist += distance(req_pos, pos[moving_agent])
            pos[moving_agent] = req_pos

    make_videos_from_images(inst_random_p, "posrandom")
    return total_dist


def use_nearest_agent_positioned_ter(inst_near_p3, k_near_p3, sites_near_p3, demands_nearest_p3, limit):
    '''Move the nearest agent
    We first position each agent at a site when the first requests arrive
        so that there will be more agents moving and sites covered at the
        same time
    If the site is too far away from start (n steps), we move another agent,
        it is useful in case it is the last request (for short sequences)'''
    total_dist = 0
    pos = [[0, 0] for _ in range(k_near_p3)]
    iteration = 0
    for dem in demands_nearest_p3:

        req_pos = sites_near_p3[dem]
        plot_grid_and_servers(inst_near_p3, iteration, grid, pos, "nearposter", total_dist)
        iteration += 1

        if req_pos not in pos:

            dist = [distance(req_pos, pos[i]) for i in range(k_near_p3)]

            if [0, 0] in pos and distance(req_pos, [0, 0]) < limit:

                moving_agent = pos.index([0, 0])
                total_dist += dist[moving_agent]

            else:

                moving_agent = dist.index(min(dist))
                total_dist += dist[moving_agent]

            pos[moving_agent] = req_pos

    make_videos_from_images(inst_near_p3, "nearposter")
    return total_dist


def random_agent(inst_random, k_random, sites_random, demands_random):
    '''Move a random agent'''

    total_dist = 0
    pos = [[0, 0] for _ in range(k_random)]
    iteration = 0

    for dem in demands_random:

        req_pos = sites_random[dem]

        plot_grid_and_servers(inst_random, iteration, grid, pos, "random", total_dist)
        iteration += 1

        # Get a random agent
        moving_agent = random.randint(0, k_random-1)

        total_dist += distance(req_pos, pos[moving_agent])
        pos[moving_agent] = req_pos

    make_videos_from_images(inst_random, "random")
    return total_dist


def random_agent_bis(inst_random2, k_random2, sites_random2, demands_random2):
    '''Move a random agent

    Slightly different : D o not move if the request is already satisfied !'''

    total_dist = 0
    pos = [[0, 0] for _ in range(k_random2)]
    iteration = 0

    for dem in demands_random2:

        req_pos = sites_random2[dem]

        plot_grid_and_servers(inst_random2, iteration, grid, pos, "randombis", total_dist)
        iteration += 1

        if req_pos not in pos:

            # Get a random agent
            moving_agent = random.randint(0, k_random2-1)

            total_dist += distance(req_pos, pos[moving_agent])
            pos[moving_agent] = req_pos

    make_videos_from_images(inst_random2, "randombis")
    return total_dist


def round_robin(inst_round, k_round, sites_round, demands_round):
    '''Each agent moves one after another

    If the sequence of moved agents is
    1, 2, 3, ..., k_round, 1
    Then the moving agent will be 2 for this request '''
    last_moved = 0
    total_dist = 0
    pos = [[0, 0] for _ in range(k_round)]
    iteration = 0

    for dem in demands_round:

        req_pos = sites_round[dem]

        plot_grid_and_servers(inst_round, iteration, grid, pos, "roundrobin", total_dist)
        iteration += 1

        if req_pos not in pos:

            # We checkwho was the last moved, and move the following one
            moving_agent = (last_moved+1) % k_round

            total_dist += distance(req_pos, pos[moving_agent])
            pos[moving_agent] = req_pos
            last_moved = moving_agent

    make_videos_from_images(inst_round, "roundrobin")
    return total_dist


def own_area(inst_own, k_own, sites_own, demands_own):
    '''Each agent watch a zone (step lines)

    If we have a 4x4 grid and 4 agents, the distribution will look like that
    |1___|
    |2___|
    |3___|
    |4___|
    '''
    total_dist = 0
    step = (grid_size//k_own)
    pos = [[0, 0] for _ in range(k_own)]
    iteration = 0

    for dem in demands_own:

        req_pos = sites_own[dem]

        plot_grid_and_servers(inst_own, iteration, grid, pos, "own", total_dist)
        iteration += 1

        # We check in whom square the request is
        moving_agent = req_pos[0]//step

        total_dist += distance(req_pos, pos[moving_agent])
        pos[moving_agent] = req_pos

    make_videos_from_images(inst_own, "own")
    return total_dist


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


def own_area_s(inst_own_s, k_own_s, sites_own_s, demands_own_s):
    '''Each agent watch a zone (squarish zone)

    If we have a 4x4 grid and 4 agents, the distribution will look like that
    |1 | 2|
    |__|__|
    |  |  |
    |3 | 4|
    '''
    total_dist = 0
    (lines, columns) = get_squarish(k_own_s)  # Get squarish decomposition of k
    l_step = grid_size//lines
    c_step = grid_size//columns
    pos = [[0, 0] for _ in range(k_own_s)]
    iteration = 0

    for dem in demands_own_s:

        req_pos = sites_own_s[dem]

        plot_grid_and_servers(inst_own_s, iteration, grid, pos, "owns", total_dist)
        iteration += 1

        # We check in whom square the request is
        moving_agent = req_pos[0]//l_step + req_pos[1]//c_step

        total_dist += distance(req_pos, pos[moving_agent])
        pos[moving_agent] = req_pos

    make_videos_from_images(inst_own_s, "owns")
    return total_dist


def popular_places(inst_pop, k_pop, sites_pop, demands_pop):
    '''Move the nearest agent
    We first position each agent at a site when the first requests arrive
        so that there will be more agents moving and sites covered at the
        same time'''

    total_dist = 0
    pos = [[0, 0] for _ in range(k_pop)]
    iteration = 0

    spots = {}

    for dem in demands_pop:

        req_pos = sites_pop[dem]

        if dem not in spots:
            spots[dem] = 1
        else:
            spots[dem] += 1

        if req_pos not in pos:
            agents_spot = [0 for _ in range(k_pop)]
            for agent in range(k_pop):
                if pos[agent] in sites_pop:
                    agents_spot[agent] = spots[sites_pop.index(pos[agent])]

            plot_grid_and_servers(inst_pop, iteration, grid, pos, "pop", total_dist)
            iteration += 1

            moving_agent = agents_spot.index(min(agents_spot))
            # print("pos =", pos, " moving=", moving_agent, " places=", spots)
            total_dist += distance(pos[moving_agent], req_pos)
            pos[moving_agent] = req_pos

    make_videos_from_images(inst_pop, "pop")
    return total_dist


def popular_places_bis(inst_pop2, k_pop2, sites_pop2, demands_pop2):
    '''Move the nearest agent
    We first position each agent at a site when the first requests arrive
        so that there will be more agents moving and sites covered at the
        same time'''

    total_dist = 0
    pos = [[0, 0] for _ in range(k_pop2)]
    iteration = 0

    spots = {}

    for dem in demands_pop2:

        req_pos = sites_pop2[dem]

        if dem not in spots:
            spots[dem] = 1
        else:
            spots[dem] += 1

        if req_pos not in pos:
            agents_spot = [0 for _ in range(k_pop2)]
            for agent in range(k_pop2):
                if pos[agent] in sites_pop2:
                    agents_spot[agent] = spots[sites_pop2.index(pos[agent])]

            plot_grid_and_servers(inst_pop2, iteration, grid, pos, "pop2", total_dist)
            iteration += 1

            # print("agents places", agents_spot)
            mini = min(agents_spot)
            less_visited = [i for i in range(k_pop2) if agents_spot[i] == mini]
            # print("less visited", less_visited)

            dist = [(i, distance(req_pos, pos[i])) for i in less_visited]
            v = min(dist, key=lambda x: x[1])

            # print(dist, v, dist.index(v))
            moving_agent, dist_moving = dist[dist.index(v)]
            # print("moving", moving_agent)
            # print("pos =", pos, " moving=", moving_agent, " places=", spots)
            total_dist += dist_moving
            pos[moving_agent] = req_pos

    make_videos_from_images(inst_pop2, "pop2")
    return total_dist


if __name__ == '__main__':

    # Get all test instances names
    for root, dirs, files in os.walk(root_dir):
        for fileName in files:
            fileSet.add(root_dir + "/" + fileName)

    results = {"one_agent": [],
               "nearest_agent": [],
               "nearest_agent_positioned": [],
               "nearest_agent_positioned_optimized": [],
               "nearest_agent_positioned_optimized2": [],
               "random_agent": [],
               "random_agent_optimized": [],
               "round_robin": [],
               "own_area": [],
               "own_area_s": [],
               "popular_places": [],
               "popular_places_optimized": [],
               "positioned_then_random": []}

    i = 0
    for file in fileSet:

        # Uncomment if you want to test on only one instance
        # if i != 0:
        #     break

        i+=1

        # Parsing file
        opt, k, sites, demands = parse_file(file)

        # Header
        print("==============")
        print("Results for file ", file)
        print("Optimal value :", opt)

        inst = file
        # Computations
        one_a = one_agent(inst, k, sites, demands)
        nearest = use_nearest_agent(inst, k, sites, demands)
        nearest_p = use_nearest_agent_positioned(inst, k, sites, demands)
        nearest_p_2 = use_nearest_agent_positioned_bis(inst, k, sites, demands)
        nearest_p_3 = use_nearest_agent_positioned_ter(inst, k, sites, demands, 180)
        random_p = positioned_then_random(inst, k, sites, demands)
        random_a = random_agent(inst, k, sites, demands)
        random_a_2 = random_agent_bis(inst, k, sites, demands)
        round_r = round_robin(inst, k, sites, demands)
        area = own_area(inst, k, sites, demands)
        area_s = own_area_s(inst, k, sites, demands)
        pop = popular_places(inst, k, sites, demands)
        pop2 = popular_places_bis(inst, k, sites, demands)

        # Verbose
        print("Use One agent algorithm (Move only one agent) :", one_a)
        print("Use nearest agent : ", nearest)
        print("Use nearest agent positioned : ", nearest_p)
        print("Use nearest agent positioned optimized : ", nearest_p_2)
        print("Use nearest agent positioned optimized2 : ", nearest_p_3)
        print("Use random agent : ", random_a)
        print("Use random agent optimized : ", random_a_2)
        print("Use round robin : ", round_r)
        print("Use own area : ", area)
        print("Use own area_s : ", area_s)
        print("Use positioned then random", random_p)
        print("Use popular places", pop)
        print("Use popular places optimized", pop2)

        # Saving results
        results["one_agent"].append(opt*100/one_a)
        results["nearest_agent"].append(opt*100/nearest)
        results["nearest_agent_positioned"].append(opt*100/nearest_p)
        results["nearest_agent_positioned_optimized"].append(opt*100/nearest_p_2)
        results["nearest_agent_positioned_optimized2"].append(opt*100/nearest_p_3)
        results["positioned_then_random"].append(opt*100/random_p)
        results["random_agent"].append(opt*100/random_a)
        results["random_agent_optimized"].append(opt*100/random_a_2)
        results["round_robin"].append(opt*100/round_r)
        results["own_area"].append(opt*100/area)
        results["own_area_s"].append(opt*100/area_s)
        results["popular_places"].append(opt*100/pop)
        results["popular_places_optimized"].append(opt*100/pop2)

    print("")

    # Ordering results
    percentages = [(algo, int(np.mean(values))) for (algo, values) in results.items()]
    percentages.sort(key=lambda x: x[1])

    # Print results
    for algo, res in percentages:
        print(algo, "has a mean performance of", res, "%")
