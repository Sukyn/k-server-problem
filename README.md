# k-server-problem

## Description

*The k-server problem is a problem of theoretical computer science in the category of online algorithms, one of two abstract problems on metric spaces that are central to the theory of competitive analysis (the other being metrical task systems). In this problem, an online algorithm must control the movement of a set of k servers, represented as points in a metric space, and handle requests that are also in the form of points in the space. As each request arrives, the algorithm must determine which server to move to the requested point. The goal of the algorithm is to keep the total distance all servers move small, relative to the total distance the servers could have moved by an optimal adversary who knows in advance the entire sequence of requests.*
(Wikipedia)

Here we design some online algorithms and evaluate their competitivity against the offline algorithms (can be computed using [1])

*[1] Marek Chrobak, Howard J. Karloff, T. H. Payne, and Sundar Vishwanathan. New results on server problems. In Proc. of SODA, 1990*

## The Algorithms

Each algorithm has at least 4 arguments
- inst :: char, instance name (for plots)
- k :: int, number of agents
- sites :: [int, int] list, the locations of places,
- demands :: int list, the places where the servers must go, in order 

- One agent algorithm : Only move one agent
- Nearest agent :  Move nearest agent
- Nearest agent positioned :  Put agents on the diagonal then move nearest agent
- Nearest agent positioned bis : Move nearest agent if every agent is not on the starting tile   
- Nearest agent positioned ter : Move nearest agent if every agent is not on the starting tile or if the target is too far from the starting tile
- Random agent :  Move a random agent
- Random agent bis :  Move a random agent if there is not any agent on the target
- Round robin :  Move agents sequentially (one after another)
- Own area :  Each agent watches some lines
- Own area bis :  Each agent watches some squares
- Positioned then random : Move random agent if every agent is not on the starting tile
- Popular places : Move the agent on the less popular spot
- Popular places bis : Move the nearest agent on one of the less popular spots

## Competitivity results

[TODO]

## Extra features

`plot_grid_and_servers` is an util function which plot the servers on the grid
`make_videos_from_images` makes a short video from plots for live presentations

## Todo :

- Competitivty analysis
- Live presentation
- Clean code
- Clean comments (some are not up-to-date)
- Add comments
- Write a one-two pages report
