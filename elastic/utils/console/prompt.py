"""
Miscellaneous tools
"""
import sys

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
    It must be "yes" (the default), "no" or None (meaning
    an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def input_multiple_lines(quit_word, prompt='> ', max_lines=0):
    """
    Input multiple lines until `quit_word` typed or 
      a max number of lines reached
    
    Args:
      quit_word
      prompt: '> '
      max_lines: if 0, no limit until quit_word is typed.
    
    Returns:
      List of each line, stripped of trailing and ending spaces.
    """
    n = 0
    lines = []
    while max_lines == 0 or n < max_lines:
        n += 1
        inp = input(prompt).strip()
        if inp == quit_word:
            break
        lines.append(inp)
    return lines
