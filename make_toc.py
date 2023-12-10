import sys


def process_line(line):
    h_size, *words = line.strip('\n').split(' ')
    indents = '  ' * (len(h_size) - 2)
    title = ' '.join(words)

    link = '-'.join(words).lower()
    link = link.replace('.', '')  # Remove '.' from the link
    link = link.replace(':', '')  # Remove ':' from the link

    toc_line = f'{indents}- [{title}](#{link})'

    return toc_line


def make_toc(filename):
    toc_lines = ['## Table of Contents']

    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            if line.startswith('#'):
                toc_line = process_line(line)
                toc_lines.append(toc_line)

    print('\n'.join(toc_lines))


if __name__ == '__main__':
    make_toc(sys.argv[1])