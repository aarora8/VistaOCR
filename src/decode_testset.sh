#!/bin/bash

#source ~/.bashrc

#source activate py35

#nvidia-smi
#echo Using GPU device: $CUDA_VISIBLE_DEVICES

export PYTHONPATH=/home/hltcoe/detter/src/VistaOCR:$PYTHONPATH

if [ $# -ne 3 ] && [ $# -ne 4 ]; then
    echo "USAGE:  ./decode_testset.sh <output-dir> <model-path> <data-path>  [<lm-path>]"
    exit 1
fi


OUTDIR=$1
model_path=$2
datadir=$3

if [ $# -eq 4 ]; then
    lmpath=$4
else
    lmpath=
fi

mkdir -p ${OUTDIR}

SCLITE=sclite
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run Decoding
python3 ${script_dir}/decode_testset.py --batch-size=32 --datadir=${datadir} --model-path=${model_path} --lm-path=${lmpath} --outdir=${OUTDIR}

if [[ $? -ne 0 ]]; then
    echo "Error in decoding"
    exit 1
fi

# Generate ref output
python ${script_dir}/print_ref.py ${datadir} > $OUTDIR/ref-chars.txt


#python ${script_dir}/normalize_arabic.py $OUTDIR/ref-chars.txt $OUTDIR/ref-chars.txt-BACKUP
#mv $OUTDIR/ref-chars.txt-BACKUP $OUTDIR/ref-chars.txt
#python ${script_dir}/normalize_arabic.py $OUTDIR/hyp-chars.txt $OUTDIR/hyp-chars.txt-BACKUP
#mv $OUTDIR/hyp-chars.txt-BACKUP $OUTDIR/hyp-chars.txt


# Squash spaces together;
python ${script_dir}/collapse_spaces.py $OUTDIR/hyp-chars.txt $OUTDIR/hyp-chars.txt.spaces
python ${script_dir}/collapse_spaces.py $OUTDIR/ref-chars.txt $OUTDIR/ref-chars.txt.spaces
mv $OUTDIR/hyp-chars.txt.spaces $OUTDIR/hyp-chars.txt
mv $OUTDIR/ref-chars.txt.spaces $OUTDIR/ref-chars.txt


# Create lowercase version too
python3 ${script_dir}/to_lower.py $OUTDIR/hyp-chars.txt $OUTDIR/hyp-chars.lower.txt
python3 ${script_dir}/to_lower.py $OUTDIR/ref-chars.txt $OUTDIR/ref-chars.lower.txt


# Turn decoding output into tokenized words
python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/hyp-chars.txt $OUTDIR/hyp-words.txt
python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/ref-chars.txt $OUTDIR/ref-words.txt

python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/hyp-chars.lower.txt $OUTDIR/hyp-words.lower.txt
python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/ref-chars.lower.txt $OUTDIR/ref-words.lower.txt


if [[ -e $OUTDIR/hyp-lm-chars.txt ]]; then
    # clean up spaces w.r.t. LM chars too
    python ${script_dir}/collapse_spaces.py $OUTDIR/hyp-lm-chars.txt $OUTDIR/hyp-lm-chars.txt.spaces
    mv $OUTDIR/hyp-lm-chars.txt.spaces $OUTDIR/hyp-lm-chars.txt

    # Also add lowercase version of lm
    python3 ${script_dir}/to_lower.py $OUTDIR/hyp-lm-chars.txt $OUTDIR/hyp-lm-chars.lower.txt

    # Now ready to get lm words
    python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/hyp-lm-chars.txt $OUTDIR/hyp-lm-words.txt
    python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/hyp-lm-chars.lower.txt $OUTDIR/hyp-lm-words.lower.txt
fi

echo "Done with decoding; Now scoring with sclite"

# Do CER measurement
${SCLITE} -r $OUTDIR/ref-chars.txt -h $OUTDIR/hyp-chars.txt -i swb -o all >/dev/null
${SCLITE} -r $OUTDIR/ref-chars.lower.txt -h $OUTDIR/hyp-chars.lower.txt -i swb -o all >/dev/null

if [[ -e $OUTDIR/hyp-lm-chars.txt ]]; then
    ${SCLITE} -r $OUTDIR/ref-chars.txt -h $OUTDIR/hyp-lm-chars.txt -i swb -o all >/dev/null
    ${SCLITE} -r $OUTDIR/ref-chars.lower.txt -h $OUTDIR/hyp-lm-chars.lower.txt -i swb -o all >/dev/null
fi

# Do WER measurement
${SCLITE} -r $OUTDIR/ref-words.txt -h $OUTDIR/hyp-words.txt -i swb -o all >/dev/null
${SCLITE} -r $OUTDIR/ref-words.lower.txt -h $OUTDIR/hyp-words.lower.txt -i swb -o all >/dev/null

if [[ -e $OUTDIR/hyp-lm-words.txt ]]; then
    ${SCLITE} -r $OUTDIR/ref-words.txt -h $OUTDIR/hyp-lm-words.txt -i swb -o all >/dev/null
    ${SCLITE} -r $OUTDIR/ref-words.lower.txt -h $OUTDIR/hyp-lm-words.lower.txt -i swb -o all >/dev/null
fi


# Now display results
# TODO -- prettier printing / display
echo "No LM CER:"
grep 'Sum/Avg' $OUTDIR/hyp-chars.txt.sys

if [ -n "$lmpath" ]; then
    echo "LM CER:"
    grep 'Sum/Avg' $OUTDIR/hyp-lm-chars.txt.sys
fi

echo "No LM WER:"
grep 'Sum/Avg' $OUTDIR/hyp-words.txt.sys

if [  -n "$lmpath" ]; then
    echo "LM WER:"
    grep 'Sum/Avg' $OUTDIR/hyp-lm-words.txt.sys
fi

