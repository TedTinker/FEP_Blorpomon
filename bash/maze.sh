#!/bin/bash -l

eval $1
eval $2
eval $3

real_arg_list=()

for arg in ${arg_list[*]}
do
    temp_file=$(mktemp)
    singularity exec maze.sif python -c "from FEP_Blorpomon.bash.slurmcraft import all_like_this; result = all_like_this('$arg'); print(result, file=open('${temp_file}', 'w'))"
    returned_value=$(cat ${temp_file})
    #real_arg_list=$(echo "${real_arg_list}${returned_value}" | jq -s 'add')
    real_arg_list=$(python -c "import json; a = json.loads('""$real_arg_list""') if '""$real_arg_list""' else []; b = json.loads('""$returned_value""') if '""$returned_value""' else []; print(json.dumps(a+b))")
    rm ${temp_file}
done

arg_list=$real_arg_list
singularity exec maze.sif python FEP_Blorpomon/bash/slurmcraft.py --comp ${comp} --agents ${agents} --arg_list "${arg_list}"
arg_list=$(echo "${arg_list}" | tr -d '[]"' | sed 's/,/, /g')
wait

echo
if [ $agents -eq 0 ]; then
    dict_jid=$(sbatch --partition=compute --wrap="sleep .1" | awk '{print $4}')
    echo "No training"
else
    jid_list=()
    max_agents=1
    for arg in ${arg_list//, / }
    do
        previous_agents=0 
        if [ $arg == "break" ]
        then
            :
        elif [ $arg == "empty_space" ]
        then
            :
        elif [ ${agents} -gt ${max_agents} ]
        then
            num_jobs=$(( ${agents} / ${max_agents} )) 
            remainder=$(( ${agents} % ${max_agents} ))
            if [ $remainder -gt 0 ]
            then
                num_jobs=$(( num_jobs + 1 ))
            else 
                remainder=${max_agents}
            fi
            for (( i=1; i<=${num_jobs}; i++ ))
            do
                if [ $i -eq ${num_jobs} ]
                then
                    agents_per_job=$(( remainder ))
                else
                    agents_per_job=$(( ${max_agents} ))
                fi
                jid=$(sbatch --export=agents_per_job=${agents_per_job},previous_agents=${previous_agents} FEP_Blorpomon/bash/main_${arg}.slurm | awk '{print $4}')
                echo "$jid : $arg ($i)"
                jid_list+=($jid)
                previous_agents=$(( previous_agents + agents_per_job ))
            done
        else
            jid=$(sbatch --export=agents_per_job=${agents},previous_agents=0 FEP_Blorpomon/bash/main_${arg}.slurm | awk '{print $4}')
            echo "$jid : $arg"
            jid_list+=($jid)
        fi
    done
fi

echo