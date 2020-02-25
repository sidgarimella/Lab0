'''Code file for baseball elimination lab created for Advanced Algorithms
Spring 2020 at Olin College. The code for this lab has been adapted from:
https://github.com/ananya77041/baseball-elimination/blob/master/src/BaseballElimination.java'''

import sys
import math
import picos as pic
import networkx as nx
import itertools
import cvxopt
import matplotlib.pyplot as plt


class Division:
    '''
    The Division class represents a baseball division. This includes all the
    teams that are a part of that division, their winning and losing history,
    and their remaining games for the season.

    filename: name of a file with an input matrix that has info on teams &
    their games
    '''

    def __init__(self, filename):
        self.teams = {}
        self.G = nx.DiGraph()
        self.readDivision(filename)

    def readDivision(self, filename):
        '''Reads the information from the given file and builds up a dictionary
        of the teams that are a part of this division.

        filename: name of text file representing tournament outcomes so far
        & remaining games for each team
        '''
        f = open(filename, "r")
        lines = [line.split() for line in f.readlines()]
        f.close()

        lines = lines[1:]
        for ID, teaminfo in enumerate(lines):
            team = Team(int(ID), teaminfo[0], int(teaminfo[1]), int(teaminfo[2]), int(teaminfo[3]), list(map(int, teaminfo[4:])))
            self.teams[ID] = team

    def get_team_IDs(self):
        '''Gets the list of IDs that are associated with each of the teams
        in this division.

        return: list of IDs that are associated with each of the teams in the
        division
        '''
        return self.teams.keys()

    def is_eliminated(self, teamID, solver):
        '''Uses the given solver (either Linear Programming or Network Flows)
        to determine if the team with the given ID is mathematically
        eliminated from winning the division (aka winning more games than any
        other team) this season.

        teamID: ID of team that we want to check if it is eliminated
        solver: string representing whether to use the network flows or linear
        programming solver
        return: True if eliminated, False otherwise
        '''
        flag1 = False
        team = self.teams[teamID]

        temp = dict(self.teams)
        del temp[teamID]

        for _, other_team in temp.items():
            if team.wins + team.remaining < other_team.wins:
                flag1 = True

        saturated_edges = self.create_network(teamID)
        if not flag1:
            if solver == "Network Flows":
                flag1 = self.network_flows(saturated_edges)
            elif solver == "Linear Programming":
                flag1 = self.linear_programming(saturated_edges)

        return flag1

    def create_network(self, teamID):
        '''Builds up the network needed for solving the baseball elimination
        problem as a network flows problem & stores it in self.G. Returns a
        dictionary of saturated edges that maps team pairs to the amount of
        additional games they have against each other.

        teamID: ID of team that we want to check if it is eliminated
        return: dictionary of saturated edges that maps team pairs to
        the amount of additional games they have against each other
        '''

        saturated_edges = {}

        self.G = nx.DiGraph()
        self.G.add_node("s")
        self.G.add_node("t")

        for otherTeamID in self.get_team_IDs():
            if otherTeamID != teamID:
                self.G.add_node(otherTeamID)
                self.G.add_edge(otherTeamID, "t", capacity=(self.teams[teamID].wins + self.teams[teamID].remaining - self.teams[otherTeamID].wins))
                for otherAgainstTeamID in self.get_team_IDs():
                    team_pair_id = str(otherTeamID) + "_" + str(otherAgainstTeamID)
                    if otherTeamID != otherAgainstTeamID and otherAgainstTeamID != teamID and self.teams[otherTeamID].get_against(other_team=otherAgainstTeamID) != 0 and not self.G.has_node('_'.join(team_pair_id.split('_')[::-1])):
                        self.G.add_node(team_pair_id)
                        saturated_edges[team_pair_id] = self.teams[otherTeamID].get_against(other_team=otherAgainstTeamID)
                        self.G.add_edge("s", team_pair_id, capacity=float(saturated_edges[team_pair_id]))
                        self.G.add_edge(team_pair_id, otherTeamID, capacity=float('inf'))
                        self.G.add_edge(team_pair_id, otherAgainstTeamID, capacity=float('inf'))

        return saturated_edges

    def network_flows(self, saturated_edges):
        '''Uses network flows to determine if the team with given team ID
        has been eliminated. You can feel free to use the built in networkx
        maximum flow function or the maximum flow function you implemented as
        part of the in class implementation activity.

        saturated_edges: dictionary of saturated edges that maps team pairs to
        the amount of additional games they have against each other
        return: True if team is eliminated, False otherwise
        '''

        flow_dict = nx.maximum_flow(self.G, "s", "t")[1]
        for t in saturated_edges:
            if float(saturated_edges[t]) != flow_dict["s"][t]:
                return True

        return False

    def linear_programming(self, saturated_edges):
        '''Uses linear programming to determine if the team with given team ID
        has been eliminated. We recommend using a picos solver to solve the
        linear programming problem once you have it set up.
        Do not use the flow_constraint method that Picos provides (it does all of the work for you)
        We want you to set up the constraint equations using picos (hint: add_constraint is the method you want)

        saturated_edges: dictionary of saturated edges that maps team pairs to
        the amount of additional games they have against each other
        returns True if team is eliminated, False otherwise
        '''

        maxflow=pic.Problem()

        f={}
        for e in self.G.edges():
            f[e]=maxflow.add_variable('f[{0}]'.format(e),1)

        F = maxflow.add_variable('F',1)
        caps = dict(nx.get_edge_attributes(self.G,'capacity'))

        maxflow.add_list_of_constraints(
          [f[e] < caps[e] for e in self.G.edges()],
          [('e',2)],
          'edges'
        )

        maxflow.add_list_of_constraints(
          [pic.sum([f[p,i] for p in self.G.predecessors(i)],'p','pred(i)') == pic.sum([f[i,j] for j in self.G.successors(i)],'j','succ(i)') 
            for i in self.G.nodes() if i not in ('s','t')],
          'i','nodes-(s,t)'
        )

        # Set source flow at s.
        maxflow.add_constraint(
          pic.sum([f[p,'s'] for p in self.G.predecessors('s')],'p','pred(s)') + F
          == pic.sum([f['s',j] for j in self.G.successors('s')],'j','succ(s)')
        )

        # Set sink flow at t.
        maxflow.add_constraint(
          pic.sum([f[p,'t'] for p in self.G.predecessors('t')],'p','pred(t)')
          == pic.sum([f['t',j] for j in self.G.successors('t')],'j','succ(t)') + F
        )

        # Enforce flow nonnegativity.
        maxflow.add_list_of_constraints(
          [f[e]>0 for e in self.G.edges()],
          [('e',2)],
          'edges'
        )

        # Set the objective.
        maxflow.set_objective('max',F)
        maxflow.solve(verbose=0,solver='glpk')

        return (int(F) < sum(saturated_edges.values()))

    def checkTeam(self, team):
        '''Checks that the team actually exists in this division.
        '''
        if team.ID not in self.get_team_IDs():
            raise ValueError("Team does not exist in given input.")

    def plotNetwork(self, ID):
        '''Plots the network using a meaningful layout.
        '''
        pos = dict()
        X = self.G.neighbors("s")
        Y = self.G.reverse(copy=True).neighbors("t")
        X = list(X)
        Y = list(Y)
        pos.update([("s", (1, int(len(list(X))/2)))])
        pos.update([("t", (4, int(len(list(Y))/2)))])
        pos.update( [(n, (2, i)) for i, n in enumerate(X)] ) 
        pos.update( [(n, (3, i)) for i, n in enumerate(Y)] )
        edge_labels = nx.get_edge_attributes(self.G,'capacity')
        nx.draw_networkx(self.G, cmap = plt.get_cmap('jet'), pos=pos, font_size=6)
        nx.draw_networkx_edge_labels(self.G, pos, labels = edge_labels, font_size=6)
        plt.title("TEAM: " + str(ID))
        plt.show()

    def __str__(self):
        '''Returns pretty string representation of a division object.
        '''
        temp = ''
        for key in self.teams:
            temp = temp + f'{key}: {str(self.teams[key])} \n'
        return temp

class Team:
    '''
    The Team class represents one team within a baseball division for use in
    solving the baseball elimination problem. This class includes information
    on how many games the team has won and lost so far this season as well as
    information on what games they have left for the season.

    ID: ID to keep track of the given team
    teamname: human readable name associated with the team
    wins: number of games they have won so far
    losses: number of games they have lost so far
    remaining: number of games they have left this season
    against: dictionary that can tell us how many games they have left against
    each of the other teams
    '''

    def __init__(self, ID, teamname, wins, losses, remaining, against):
        self.ID = ID
        self.name = teamname
        self.wins = wins
        self.losses = losses
        self.remaining = remaining
        self.against = against

    def get_against(self, other_team=None):
        '''Returns number of games this team has against this other team.
        Raises an error if these teams don't play each other.
        '''
        try:
            num_games = self.against[other_team]
        except:
            raise ValueError("Team does not exist in given input.")

        return num_games

    def __str__(self):
        '''Returns pretty string representation of a team object.
        '''
        return f'{self.name} \t {self.wins} wins \t {self.losses} losses \t {self.remaining} remaining'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        division = Division(filename)
        for (ID, team) in division.teams.items():
            print(f'{team.name}: Eliminated? {division.is_eliminated(team.ID, "Linear Programming")}')
            # division.plotNetwork(ID)
    else:
        print("To run this code, please specify an input file name. Example: python baseball_elimination.py teams2.txt.")
