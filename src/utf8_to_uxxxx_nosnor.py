import sys
import textutils


input_file = sys.argv[1]

with open(input_file, 'r') as fh:

    for line in fh:
        line_utf8 = textutils.utf8_to_uxxxx(line.strip())

        print("%s\n" % line_utf8)

