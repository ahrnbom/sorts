# This is supposed to run inside of the Singularity container. For an example of
# building and starting the container, see run_singularity.sh

echo "> Input: $@"

if [ "$1" = "bash" ]; then   
    echo "> Booting into bash inside Singularity..."
    export PROMPT_COMMAND="echo -n \[\ Singularity \]\ "
    exec bash
else
    echo "> Executing your input..."
    exec "$@"
fi
