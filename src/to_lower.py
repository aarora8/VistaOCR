import sys

input_file = sys.argv[1]
output_file = sys.argv[2]



def uxxxx_to_utf8(in_char):
    # First get the 'xxxx' part out of the current 'uxxxx' char
    char = in_char[1:]
    # Now decode the hex code point into utf-8
    utf8_char = chr(int(char, 16))
    return utf8_char

def utf8_to_uxxxx(in_char):
    char_hex = hex(ord(in_char))[2:].zfill(4).lower()
    uxxxx_char = "u%s" % char_hex
    return uxxxx_char


output_lines = []
with open(input_file, "r") as fh:
    for line in fh:
        utt, uttid = line.strip().split("(")
        uttid = uttid.strip(")")

        char_array = utt.split()

        for i in range(len(char_array)):
            utf8_char = uxxxx_to_utf8(char_array[i])
            char_array[i] = utf8_to_uxxxx(utf8_char.lower())
            

        new_utt = ' '.join(char_array)
        output_lines.append(("%s (%s)\n" % (new_utt, uttid)))

with open(output_file, "w") as fh:
    for line in output_lines:
        fh.write(line)
