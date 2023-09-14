#!/usr/bin/env python3
import pprint, argparse, pickle, json

import maze 

# ------------------------------
# For every part and every map we run the corresponding algorithm
#   Some credit is given for finding a correct solution
#   Some credit is given for finding an optimal solution
#   Some credit is given for not exploring too many states 
#
# Credit will be given for visible test cases and hiddle ones.
#   -Parts-         -Visible-                       -Hidden-
#   BFS             (30) Tiny, Small, Open          (20) Medium, Large
#   AStar_single    (30) Tiny, Small, Open          (20) Medium, Large
# ------------------------------

def fail(message):
    return {
        'score'     : 0,
        'output'    : message, 
        'visibility': 'visible', 
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description     = 'CS440 MP5 Autograder', 
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--gradescope', default = False, action = 'store_true',
                        help = 'save output in gradescope-readable json file')

    arguments   = parser.parse_args()
    
    try:
        import submitted 
    except ImportError:
        message = 'could not find module \'submitted\', did you upload `submitted.py`?'
        if arguments.gradescope:
            with open('results.json', 'w') as file:
                file.write(json.dumps(fail(message)))
        else:
            print(message)
        raise SystemExit

def generate_answer_key(path, mazes, solutions):
    key_instructor  = tuple({case: (getattr(submitted, solution)(maze), maze.states_explored)
        for case, maze in mazes.items()}
        for mazes, solution in zip(mazes, solutions))
    key_student     = tuple({case: (len(sol[0]), sol[1]) for case, sol in part.items()} 
        for part in key_instructor)
    pickle.dump(key_instructor, open(path['instructor'], 'wb'))
    pickle.dump(key_student,    open(path['student'],    'wb'))

def load_answer_key(path):
    try:
        return pickle.load(open(path['instructor'], 'rb'))
    except FileNotFoundError:
        print('running in student mode (instructor key unavailable)')
        return pickle.load(open(path['student'],    'rb'))

def grade_optimal(name, key, mazes, solution, weight = 1):
    def grade(case, maze):
        path, states_explored = key[case]
        z = getattr(submitted, solution)(maze)
        # check that the path is valid 
        ret_valid = maze.validate_path(z)
        score_validity  = int(ret_valid is None)
        # check that the length of the student’s path matches 
        true_len = (path if type(path) is int else len(path))
        if score_validity:
            score_length    = int(len(z) == true_len)
            # check that student explores at most 10% more states than solution
            score_explored = maze.states_explored < 1.1 * states_explored
        else:
            score_length = 0
            score_explored = 0
        return (
            {
                'name'      : '{0}: `validate_path(_:)` for \'{1}\' maze'.format(name, case),
                'output'    : 'Your path is valid' if score_validity else "Your path is not valid, error: {}".format(ret_valid),
                'score'     : 2 * weight * score_validity * 2,
                'max_score' : 2 * weight * 2,
                'visibility': 'visible'
            },
            {
                'name'      : '{0}: not too many states explored for \'{1}\' maze'.format(name, case),
                'output'    : 'You explored {} states, you should explore fewer than 1.1 * {}'.format(maze.states_explored, states_explored),
                'score'     : weight * score_explored * 2,
                'max_score' : weight * 2,
                'visibility': 'visible'
            },
            {
                'name'      : '{0}: correct path length for \'{1}\' maze'.format(name, case),
                'output'    : 'Your path length is {}, the correct length is {}'.format(len(z), true_len),
                'score'     : 2 * weight * score_length * 2,
                'max_score' : 2 * weight * 2,
                'visibility': 'visible'
            },
        )
            
    return tuple(item for case, maze in mazes.items() for item in grade(case, maze))

# for extra credit part
def grade_suboptimal(name, key, mazes, solution):
    def grade(case, maze):
        path, states_explored = key[case]
        z = getattr(submitted, solution)(maze)
        # check that the path is valid 
        ret_valid = maze.validate_path(z)
        score_validity  = int(ret_valid is None)
        # check that the path length isn't too bad 
        sol_len = (path if type(path) is int else len(path))
        if score_validity:
            score_length    = ( len(z)  < 1.2 * sol_len )
            
            score_explored = maze.states_explored < 1.2 * states_explored
        else:
            score_length    = 0
            score_explored = 0

        return (
            {
                'name'      : '{0}: `validate_path(_:)` for \'{1}\' maze'.format(name, case),
                'output'    : 'Your path is valid' if score_validity else "Your path is not valid, error: {}".format(ret_valid),
                'score'     : 1 * score_validity/4,
                'max_score' : 1/4,
                'visibility': 'visible'
            },
            {
                'name'      : '{0}: not too many states explored for \'{1}\' maze'.format(name, case),
                'output'    : 'You explored {} states, you should explore fewer than 1.2 * {}'.format(maze.states_explored, states_explored),
                'score'     : 2 * score_explored/4,
                'max_score' : 2/4,
                'visibility': 'visible'
            },
            {
                'name'      : '{0}: correct path length for \'{1}\' maze'.format(name, case),
                'output'    : 'Your path length is {}, it should be less than 1.2 * {}'.format(len(z), sol_len),
                'score'     : 2 * score_length/4,
                'max_score' : 2/4,
                'visibility': 'visible'
            },
        )
            
    return tuple(item for case, maze in mazes.items() for item in grade(case, maze))

def main():    
    solutions = ('bfs', 'astar_single', 'astar_multiple')
    # solutions = ('bfs', 'astar_single', 'astar_multiple', 'fast')
    for solution in solutions:
        if not hasattr(submitted, solution):
            return fail('module \'submitted\' is missing expected member \'{0}\''.format(solution))
        if not callable(getattr(submitted, solution)):
            return fail('member \'{0}\' in module \'submitted\' is not callable'.format(solution))
    
    mazes = (
        # part 1 (BFS): 50 points total, 10 points per case
        {case: maze.Maze('data/part-1/{0}'.format(case))
            for case in ('tiny', 'small', 'open')}, # 'medium', 'large',
        # part 2 (astar_single): 50 points total, 10 points per case 
        {case: maze.Maze('data/part-2/{0}'.format(case))
            for case in ('tiny', 'small', 'open')}, # 'medium', 'large',
        # part 3 (astar_multi): 10 points total, 1.25 points per case 
        {case: maze.Maze('data/part-3/{0}'.format(case))
            for case in ('tiny', 'open', 'tiny_c', 'one_d')}, # 'cross', 'small', 'medium', 'corner'
    )
    

    # generate_answer_key({'instructor': 'key_i', 'student': 'key_s'}, mazes, solutions)
    key             = load_answer_key({'instructor': 'key_i', 'student': 'key_s'})
    first_parts    = tuple(item for i, points in zip(range(0, 2), (1, 1, 1))
        for item in grade_optimal('part-{0}'.format(i + 1), key[i], mazes[i], solutions[i], 
            weight = points))
    last_part      = tuple(item for i in range(2, 3) for item in grade_suboptimal('extra credit-part-{0}'.format(i + 1), key[i], mazes[i], solutions[i]))
    
    # construct grade dictionary for gradescope 
    return {
        'visibility': 'visible', 
        #'tests': first_parts
        # main parts + extra credit
        # 'tests': first_parts + last_part
        'tests': first_parts + last_part
    } 
    
if __name__ == "__main__":
    results     = main()
    if arguments.gradescope:
        with open('results.json', 'w') as file:
            file.write(json.dumps(results))
    else:
        pprint.pprint(results)
    
