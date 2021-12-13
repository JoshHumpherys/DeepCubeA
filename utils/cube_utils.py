from typing import List


def generate_3qtm_cube_moves() -> List[str]:
    moves: List[str] = ["%s%s" % (f, n) for f in ['U', 'D', 'L', 'R', 'B', 'F'] for n in ['', '\'']]
    moves3 = set()

    def moves_are_parallel(m1, m2):
        return \
            m1 == 'U' and m2 == 'D' or m1 == 'D' and m2 == 'U' or \
            m1 == 'L' and m2 == 'R' or m1 == 'R' and m2 == 'L' or \
            m1 == 'B' and m2 == 'F' or m1 == 'F' and m2 == 'B'

    for i in range(len(moves)):
        for j in range(len(moves)):
            for k in range(len(moves)):
                if len(moves[i]) != len(moves[j]) and moves[i][0] == moves[j][0]:
                    moves3.add(moves[k])
                elif len(moves[j]) != len(moves[k]) and moves[j][0] == moves[k][0]:
                    moves3.add(moves[i])
                elif len(moves[i]) != len(moves[k]) and moves[i][0] == moves[k][0] and \
                        moves_are_parallel(moves[i][0], moves[j][0]):
                    moves3.add(moves[j])
                elif moves[i] == moves[j] and moves[j] == moves[k]:
                    moves3.add(moves[i][0] if len(moves[i]) == 2 else moves[i] + '\'')
                else:
                    moves3.add('%s %s %s' % (moves[i], moves[j], moves[k]))

    for i in range(len(moves)):
        for j in range(len(moves)):
            if len(moves[i]) == len(moves[j]) or moves[i][0] != moves[j][0]:
                moves3.add('%s %s' % (moves[i], moves[j]))

    return list(moves3)


def reverse_alg_string(alg_string: str) -> str:
    moves = alg_string.split(' ')
    rev_moves = []
    for move in moves:
        rev_moves.append(move[0] if len(move) == 2 else move + '\'')

    rev_moves.reverse()

    return ' '.join(rev_moves)
