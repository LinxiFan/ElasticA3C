"""
Colored output on console.
"""
import re
import os


CONSOLE_STYLE = dict(
    bold=1, b=1,
    dark=2, d=2,
    italic=3, i=3,
    underline=4, u=4,
    blink=5, l=5,
    reverse=7, r=7,
    hidden=8, h=8,
    strikethrough=9, s=9
)


CONSOLE_COLOR = dict(
    gray=30,
    grey=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
)

_available_colors = list(CONSOLE_COLOR)
_available_styles = list(CONSOLE_STYLE)


def colorize(text, color=None, styles=[], background=None):
    """Colorize text display in console.
    
    Args:
      text (str): to be colorized
      color (str): gray, red, green, yellow, blue, magenta, cyan, white
      styles (list or tuple): bold (b), dark (d), italic (i), underline (u),
          blink (l), reverse (r), hidden (h), strikethrough (s)
      background (str): gray, red, green, yellow, blue, magenta, cyan, white

    Returns:
      str: colorized string

    Warning:
      Some of the styles and colors might not display in certain terminals.

    """
    if os.getenv('ANSI_COLORS_DISABLED') is not None:
        return text
    
    color_header = ''
    fmt_str = '\033[{}m'

    if color:
        assert color in _available_colors, \
            '{} is not a valid color. [{}]'.format(color, _available_colors) 
        color_header += fmt_str.format(CONSOLE_COLOR[color])

    if background:
        assert background in _available_colors, \
            '{} is not a valid background. [{}]'.format(background, _available_colors) 
        color_header += fmt_str.format(CONSOLE_COLOR[background] + 10)

    if styles:
        for style in styles:
            assert style in CONSOLE_STYLE, \
                '{} is not a valid style. [{}]'.format(style, _available_styles)
            color_header += fmt_str.format(CONSOLE_STYLE[style])

    CONSOLE_RESET = '\033[0m' # WARNING: cannot be r'string'.
    return color_header + text + CONSOLE_RESET


def cprint(text, color=None, styles=[], background=None, **kwargs):
    """Prints colorized text.

    Args:
      color: same as ``colorize()``
      styles: same as ``colorize()``
      background: same as ``colorize()``
      **kwargs: kwargs for print()
    """
    print((colorize(text, color=color, styles=styles, background=background)), **kwargs)


def _replace_n(s, spans, replacements):
    # replace all spans (list of tuples) with new substrings
    assert len(spans) == len(replacements)
    if not spans:
        return s
    new_s = s[:spans[0][0]]
    for i in range(len(spans) - 1):
        assert spans[i][1] <= spans[i+1][0]
        new_s += replacements[i] + s[spans[i][1]: spans[i+1][0]]
    return new_s + replacements[-1] + s[spans[-1][1]:]
    
    
# match colorize syntax [color,configs...]`string`
_color_re = re.compile(r'\[([^\]]+)\]`([^`]+)`')

def color_format(string):
    """Colorize a string with special formatting syntax.
    
    Syntax [color,styles,background]`string to be colorized`
    styles should be represented as single-char shorthands, 
    e.g. ``bil`` means bold and italic and blinking text.
    
    Example:
        ::

            This is [blue]`blue` text with [red,bu]`bold underlined red` 
            This is [green,il,yellow]`green italic blinking text on yellow` 
            Square bracket[yoyo][magenta,,green]`vanilla magenta text on green`
    """
    spans = []
    replacements = []
    for match in _color_re.finditer(string):
        color_config, text = match.groups()
        color_config = color_config.strip().split(',')
        assert len(color_config) <= 3, 'Invalid color config: ' + color_config
        color_config.extend([None] * (3 - len(color_config))) # pad to length 3
        color, styles, background = color_config
        
        spans.append(match.span())
        colored_text = colorize(text, color=color, 
                                styles=set(styles) if styles else None, 
                                background=background)
        print(colored_text)
        replacements.append(colored_text)
    return _replace_n(string, spans, replacements)