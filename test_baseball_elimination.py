'''Test file for baseball elimination lab created for Advanced Algorithms
Spring 2020 at Olin College.'''

from baseball_elimination import Division


def assert_eliminated(division, team):
    '''Asserts that the team in the given division is eliminated using both
    the linear programming and the network flows methodology for solving.
    Prints a failure message if they are not.
    '''
    try:
        assert division.is_eliminated(team.ID, "Linear Programming") == True
    except AssertionError:
        print("Failure in linear programming is eliminated: " + team.name + " should be eliminated.")
    try:
        assert division.is_eliminated(team.ID, "Network Flows") == True
    except AssertionError:
        print("Failure in network flows is eliminated: " + team.name + " should be eliminated.")

def assert_not_eliminated(division, team):
    '''Asserts that the team in the given division is not eliminated using both
    the linear programming and the network flows methodology for solving.
    Prints a failure message if the program returns that they are.
    '''
    try:
        assert division.is_eliminated(team.ID, "Linear Programming") == False
    except AssertionError:
        print("Failure in linear programming is eliminated: " + team.name + " should NOT be eliminated.")
    try:
        assert division.is_eliminated(team.ID, "Network Flows") == False
    except AssertionError:
        print("Failure in network flows is eliminated: " + team.name + " should NOT be eliminated.")

def test_teams2():
    '''Runs all test cases on the input matrix that can be found in teams2.txt.
    '''
    division = Division("teams2.txt")
    for (ID, team) in division.teams.items():
        if team.name == "NewYork" or team.name == "Baltimore" or team.name == "Boston" or team.name == "Toronto":
            assert_not_eliminated(division, team)
        elif team.name == "Detroit":
            assert_eliminated(division, team)

def test_teams4():
    '''Runs all test cases on the input matrix that was given in the lab
    description. It is stored in teams4.txt.
    '''
    division = Division("teams4.txt")
    for (ID, team) in division.teams.items():
        if team.name == "Prava" or team.name == "Vicky":
            assert_eliminated(division, team)
        elif team.name == "Emily" or team.name == "Shashank":
            assert_not_eliminated(division, team)

if __name__ == '__main__':
    test_teams2()
    test_teams4()
    print("All tests have completed.")
