import qiskit
import glob
import os
from math import floor
from copy import deepcopy
import warnings

#import pydot
from collections import defaultdict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter
from qiskit.quantum_info.analysis import hellinger_fidelity   

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit import visualization
from itertools import permutations


#from qiskit.pulse import Schedule, Gaussian, SamplePulse, DriveChannel, ControlChannel
from qiskit import IBMQ, transpile, schedule as build_schedule


def return_synth_micro_bench(option, 
                             delay_len, 
                             q0_in = 1, 
                             q1_in = 1, 
                             slide_gate = 'u3',
                             gate_params = [np.pi,0,np.pi],
                            init_entangle = True):
    '''
    Return a benchmark with 
    selected option = {'swap_interfere', 'cx_interfere', 'no_interfere' ,'double_window'}
    
    delay_len = length of delay line in terms of ID gates inserted. This should be an even number
    
    q0_in = val of q0, default 1
    
    q1_in = val of q1, default 1
    
    slide_gate = gate to use for experiments, default u3, but should be u3,u2,x,y
    
    gate_params = gate parameters to use with sliding gate, defalult [np.pi,0,np.pi]
    
    init_entangle = will put first two qubits in EPR state if True else place CX. default True
    '''
    if delay_len%2 != 0:
        print('Please use a correct delay line value.')
        return 0
    
    qc = QuantumCircuit(3,3)
    if q0_in == 1:
        qc.x(0)
    if q1_in == 1:
        qc.x(1)
        
    
    if init_entangle == True:
        qc.h(0)
        qc.cx(0,1)
    else:
        qc.cx(0,1)
    
    if option == 'swap_interfere':
        qc.cx(1,2)
        qc.cx(2,1)
        qc.cx(1,2)
        
    if option == 'double_window':
        qc.cx(1,2)
        
    for i in range(0,delay_len):
        
        if option == 'double_window':
            qc.id(2)
        else:
            qc.id(1)
        if i == (delay_len-2)/2 and option == 'cx_interfere':
            for j in range(0,2):
                qc.cx(1,2)
    
    
    if slide_gate == 'u3':
        qc.u3(gate_params[0],gate_params[1],gate_params[2],0)
    
    elif slide_gate == 'u2':
        qc.u2(gate_params[0],gate_params[1],0)
    
    elif slide_gate == 'x':
        qc.x(0)
    
    elif slide_gate == 'y':
        qc.y(0)
        
    else:
        print('Please use correct type of sliding gate.')
        return 0
    
    if option == 'swap_interfere':
        qc.cx(1,2)
        qc.cx(2,1)
        qc.cx(1,2)
    
    if option == 'double_window':
        if slide_gate == 'u3':
            qc.u3(gate_params[0],gate_params[1],gate_params[2],1)
    
        elif slide_gate == 'u2':
            qc.u2(gate_params[0],gate_params[1],1)
    
        elif slide_gate == 'x':
            qc.x(1)
    
        elif slide_gate == 'y':
            qc.y(1)
        
        qc.cx(1,2)
        
    if init_entangle == True:
        qc.cx(0,1)
        qc.h(0)
    else:
        qc.cx(0,1)
    
    
    qc.measure(0,0)
    qc.measure(1,1)
    
    return qc
    
   
    

def get_circ_and_transpile(file_name,backend,optimization_level=3, initial_layout = None,mapper=None):
    '''
    gets circuit from file and transpiles it for a backend
    '''
    f = open(file_name).read()
    qc = QuantumCircuit().from_qasm_str(f)
    qc = transpile(qc,backend,optimization_level=optimization_level,initial_layout=initial_layout,layout_method=mapper)
    return qc



def delay_vect_static_val(dimension,location):
    return [location]*dimension


def count_id_needed(backend,qubit,start_t,end_t, location=0.5, output_in_dt=True):
    '''
    counts the number of id gates needed to insert within a delay
    window at a specified location (i.e. location = 0.5 is halfway in idle window)
    '''
    if start_t == 0:
        no_id_gates = 0
        
    else:
        if output_in_dt == True:
            cals = backend.defaults().instruction_schedule_map
            id_time = cals.get('id', qubit, *[]).duration
            total_delay = end_t - start_t
            no_id_gates = floor((total_delay*location)/id_time)
        
        else:
            properties = backend.properties()
            id_time = properties.gate_length('id',qubit)
            total_delay = end_t - start_t
            no_id_gates = floor((total_delay*location)/id_time)
    
    return no_id_gates

def count_id_for_window(backend,qubit,start_t,end_t, output_in_dt=True):

    if start_t == 0:
        no_id_gates = 0
        
    else:
        if output_in_dt == True:
            cals = backend.defaults().instruction_schedule_map
            id_time = cals.get('id', qubit, *[]).duration
            total_delay = end_t - start_t
            no_id_gates = floor((total_delay)/id_time)
        
        else:
            properties = backend.properties()
            id_time = properties.gate_length('id',qubit)
            total_delay = end_t - start_t
            no_id_gates = floor((total_delay)/id_time)
    
    return no_id_gates

    
def print_gate_info(gate_instruction):
    print('Gate: %s | Params %s | Qubit: %s'% (gate_instruction[0].name,gate_instruction[0].params,gate_instruction[1][0].index))


    
    
def fake_gate(qc0):
    qc = QuantumCircuit(qc0.num_qubits,qc0.num_clbits)
    for item in qc0:
        if item[0].name == 'cx':
            local_qubits = []
            for thing in item[1]:
                local_qubits.append(thing.index)
            for thing in local_qubits:
                qc.rz(0,thing)
            qc.append(item[0],item[1],item[2])
        else:
            qc.append(item[0],item[1],item[2])
    return qc
    
    
def create_circuit_w_delay_id(circuit,id_inst_label, id_inst_count):
    qc = QuantumCircuit(circuit.num_qubits,circuit.num_clbits)
    
    for item in circuit:
        if item[0].label in id_inst_label:

            index_val = id_inst_label.index(item[0].label)
            
            #debugging..can comment out later
            
            #print_gate_info(item)
            #print(f'id count:{id_inst_count[index_val]}')
            #print(f'label: {id_inst_label[index_val]}')
            
            qc.append(item[0],item[1],item[2])
            for i in range(0,id_inst_count[index_val]):
                qc.id(item[1])
                
        else:
            #print(qc.qasm())
            qc.append(item[0],item[1],item[2])
    
    #quick workaround to remove label
    
    for item in qc:
        item[0].label = None
    
    return qc

def create_circuit_w_delay_id_full_padding(circuit,id_inst_label, id_inst_count, id_total_window):
    qc = QuantumCircuit(circuit.num_qubits,circuit.num_clbits)
    
    
    for item in circuit:
        if item[0].label in id_inst_label:

            index_val = id_inst_label.index(item[0].label)
            
            #debugging..can comment out later
            
            #print_gate_info(item)
            #print(f'id count:{id_inst_count[index_val]}')
            #print(f'label: {id_inst_label[index_val]}')
            
            qc.append(item[0],item[1],item[2])
            for i in range(0,id_inst_count[index_val]):
                qc.id(item[1])
                
        else:
            #print(qc.qasm())
            qc.append(item[0],item[1],item[2])
    
    #we have rescheduled circuit in qc. now we need to pad it!
    
    #information needed to pad window
    qubits = [None]*len(id_inst_label)
    id_labels_padding = [None]*len(id_inst_label)
    id_padding_counts = []
    
    for i in range(len(id_inst_count)):
        id_padding_counts.append(id_total_window[i]-id_inst_count[i])
    
    #determine labels for where id padding should go
    for i in range(len(circuit)):
        #print(f'label:{circuit[i][0].label}')
        if circuit[i][0].label in id_inst_label:
            index_val = id_inst_label.index(circuit[i][0].label)
            #print(index_val)
            qubits[index_val] = circuit[i][1][0].index
            gate_found = False
            
            for j in range(i,-1,-1):
                if gate_found == True:
                    break
                
                local_qb = []
                for k in circuit[j][1]:
                    local_qb.append(k.index)
                
                #print(local_qb)

                if (circuit[j][0].name == 'cx') and qubits[index_val] in local_qb:
                    id_labels_padding[index_val] = circuit[j][0].label
                    gate_found = True
    
    #print(id_padding_counts)
    #print(qubits)
    #print(id_labels_padding)
    
    #create fully padded circuit 
    qc_padded = QuantumCircuit(circuit.num_qubits,circuit.num_clbits)
    for item in qc:
        if item[0].label in id_labels_padding:

            index_val = id_labels_padding.index(item[0].label)

            
            qc_padded.append(item[0],item[1],item[2])
            for i in range(0,id_padding_counts[index_val]):
                qc_padded.id(qubits[index_val])
                
        else:
            qc_padded.append(item[0],item[1],item[2])
    
    
    #quick workaround to remove label
    for item in qc_padded:
        item[0].label = None
    
    return qc_padded


def create_circuit_w_delay_gate(circuit,id_inst_label, delay_gate_vals,output_in_dt):
    qc = QuantumCircuit(circuit.num_qubits,circuit.num_clbits)
    

    
    for item in circuit:
        if item[0].label in id_inst_label:
            index_val = id_inst_label.index(item[0].label)
            qc.append(item[0],item[1],item[2])
            if output_in_dt == True:
                qc.delay(delay_gate_vals[index_val], item[1],unit='dt')
            if output_in_dt == False:
                qc.delay(delay_gate_vals[index_val], item[1],unit='s')
                
        else:
            #print(qc.qasm())
            qc.append(item[0],item[1],item[2])
    
    #quick workaround to remove label
    for item in qc:
        item[0].label = None
    
    return qc



def return_adj_sched_circuit(qc, backend=None, output_in_dt=True, location = 0.5, use_padding=False, fixed_placement = True, use_delay = False, return_verbose =False,DELAY_VECT = [0]*1000):
    '''
    move single qubit gates in circuit from alap to adjusted schedule
    returns - 
        qc_new - the adjusted quantum circuit
        
    warning - unexpected results if basis gates are not used for a backend!
    '''


    if backend is None:
        warnings.warn("Backend needed to produce timing information.")
    
    basis_gates = backend.configuration().basis_gates
    dt = 1
    
    circuit = deepcopy(qc)
    
    for i in range(0,len(circuit)):
        try:
            circuit[i][0].label = str(i)
        except:
            continue
            
    if output_in_dt:
        #use time in dt
        dt = backend.configuration().dt
        cals = backend.defaults().instruction_schedule_map
        
    else:
        #use time in seconds
        gate_times = backend.properties().gate_length

    free_times = defaultdict(list)
    curr_time = defaultdict(int)
    visited_qubits = []
        
    #holds visited qubits and associated labels...will be used to identify where to put id gates
    qubit_tracker = []
    id_inst_label = []
    
    timing_for_counts = []
    
    
        
    for i in range(len(circuit.data)-1,-1,-1):
        inst = circuit.data[i][0]
        label = inst.label
        qubits = circuit.data[i][1]
        clbits = circuit.data[i][1]
        if inst.name not in basis_gates:
            #commented out below. now will break with improper use
            #continue
            if (inst.name != "measure") and (inst.name != "barrier"):
                warnings.warn("Use basis set for selected backend.")
                
            else:
                continue
        qubit_indices = tuple(qubit.index for qubit in qubits)
        for item in qubit_indices:
            if item not in visited_qubits:
                visited_qubits.append(item)
        start_time = max(curr_time[q] for q in qubit_indices)
        for q in qubit_indices:
            if start_time > curr_time[q]:
                free_times[q].append([curr_time[q], start_time])
               
                for j in range(len(qubit_tracker)-1,-1,-1):
                    if qubit_tracker[j][0] == q :
                    #if qubit_tracker[j][0] == q:
                        #added below
                        if qubit_tracker[j][2] =='cx':
                            break
                        
                        temp_label = qubit_tracker[j][1]
                        qubit_number = qubit_tracker[j][0]
                        end_group_found = False
                        
                        for k in range(len(circuit.data)):
                            if circuit.data[k][0].label == temp_label:
                                temp_circ_index = k
                                local_qubits = tuple(qubit.index for qubit in circuit.data[temp_circ_index][1])
                                stop_gates = ['cx','barrier','measure']

                                
                                while ((circuit.data[temp_circ_index][0].name not in stop_gates) and (qubit_number in local_qubits)) :
                                    temp_circ_index = temp_circ_index + 1
                                    local_qubits = tuple(qubit.index for qubit in circuit.data[temp_circ_index][1])
                                    
                                
                                end_group_found = True
                                temp_label = circuit.data[temp_circ_index-1][0].label
                                
                                
                            if end_group_found ==True:
                                break
                         
                        id_inst_label.append(temp_label)
                        #old
                        #id_inst_label.append(qubit_tracker[j][1])
                        
                        
                        
                        
                        timing_for_counts.append((q,curr_time[q],start_time))
                        break
        
        
        if output_in_dt:
            duration = cals.get(inst.name, qubit_indices, *inst.params).duration
        else:
            duration = gate_times(inst.name,qubit_indices)
        for q in qubit_indices:
            new_time = start_time + duration
            if output_in_dt:
                new_time = int(new_time)
            curr_time[q] = new_time
            
            #edited this
            #add qubits to tracker if single qubit gate
            #if inst.name != 'cx':
            #    qubit_tracker.append([q,label])
            qubit_tracker.append([q,label,inst.name])
            
    end_time = max(curr_time.values())
    
    
    if use_delay == False:
        if fixed_placement == True:
            #if all gates moved to same point
            delay_vect = delay_vect_static_val(len(id_inst_label),location)

            id_inst_count = [] 
            id_total_window = []
            for i in range(0,len(timing_for_counts)):
                id_inst_count.append(count_id_needed(backend,
                                                     timing_for_counts[i][0],
                                                     timing_for_counts[i][1],
                                                     timing_for_counts[i][2],
                                                     location=delay_vect[i]
                                                     ,output_in_dt=output_in_dt))
                id_total_window.append(count_id_for_window(backend,
                                                     timing_for_counts[i][0],
                                                     timing_for_counts[i][1],
                                                     timing_for_counts[i][2],
                                                     output_in_dt=output_in_dt))
        else:
            #optimized_vect_search()
            #GOKUL = Heavily Modified
            delay_vect = DELAY_VECT

            id_inst_count = [] 
            id_total_window = []
            other_count = []
            for i in range(0,len(timing_for_counts)):
                id_inst_count.append(count_id_needed(backend,
                                                     timing_for_counts[i][0],
                                                     timing_for_counts[i][1],
                                                     timing_for_counts[i][2],
                                                     location=delay_vect[i]
                                                     ,output_in_dt=output_in_dt)) #GOKUL 
                
                id_total_window.append(count_id_for_window(backend,
                                                     timing_for_counts[i][0],
                                                     timing_for_counts[i][1],
                                                     timing_for_counts[i][2],
                                                     output_in_dt=output_in_dt))
                
                other_count.append(timing_for_counts[i])
            #return 0


        
        if use_padding == False:
            qc_new = create_circuit_w_delay_id(circuit,id_inst_label, id_inst_count)
        if use_padding == True:
            qc_new = create_circuit_w_delay_id_full_padding(circuit,id_inst_label, id_inst_count,id_total_window)

    
    elif use_delay == True:
        if fixed_placement == True:
            #if all gates moved to same point
            delay_vect = delay_vect_static_val(len(id_inst_label),location)
            
            delay_gate_vals = []
            for i in range(0,len(timing_for_counts)):
                delay_gate_vals.append((timing_for_counts[i][2]-timing_for_counts[i][1])*delay_vect[i])
                
            qc_new = create_circuit_w_delay_gate(circuit,id_inst_label, delay_gate_vals,output_in_dt)
            
        else:
            optimized_vect_search()
            return 0

        
        
    if return_verbose == True:
        
        return qc_new, timing_for_counts, id_inst_label
    else:
        return qc_new, timing_for_counts
     
    

def optimized_vect_search():
    print("this feature is in progress. delay vector optimization here.")
    return 0


def hist_from_counts(counts):
    lists = sorted(counts.items()) 
    x, y = zip(*lists) 
    shots = sum(y)
    vals = [i/shots for i in y]
    f=plt.figure(figsize=(12, 8), dpi=80)
    plt.bar(x, vals)
    plt.ylabel('probabilities')
    plt.show()

    
def return_max_index(values):
    return values.index(max(values))

def print_metrics(values, placement_points):
    print(f'Max fidelity: {max(values)}, index: {values.index(max(values))}')
    print(f'Placement point: {placement_points[values.index(max(values))]} before ALAP\n')
                                                 
    print(f'Min fidelity: {min(values)}, index: {values.index(min(values))}')
    print(f'Placement point: {placement_points[values.index(min(values))]} before ALAP\n')
    
    print(f'Delta between max and min: {max(values)-min(values)}\n')

def find_error(sim_counts,default_counts,new_counts):
    correct_key = max(sim_counts,key=sim_counts.get)
    error1 = (sim_counts[correct_key]-default_counts[correct_key])/sim_counts[correct_key]
    error2 = (sim_counts[correct_key]-new_counts[correct_key])/sim_counts[correct_key]
    print(f'Error between default simulation and default: {error1}')
    print(f'Error between default simulation and rescheduled: {error2}')


#original ALAP timing

def find_idle_time_alap(circuit, backend=None, output_in_dt=True):
    """
    Return a dictionary from qubit to idle times. Uses ALAP
    """

    if backend is None:
        warnings.warn("Backend needed to produce timing information.")
    
    basis_gates = backend.configuration().basis_gates
    dt = 1
    if output_in_dt:
        #use time in dt
        dt = backend.configuration().dt
        cals = backend.defaults().instruction_schedule_map

        free_times = defaultdict(list)
        curr_time = defaultdict(int)
        visited_qubits = []


        for i in range(len(circuit.data)-1,-1,-1):
            inst = circuit.data[i][0]
            qubits = circuit.data[i][1]
            clbits = circuit.data[i][1]

            if inst.name not in basis_gates:
            #commented out below. now will break with improper use

                if (inst.name != "measure") and (inst.name != "barrier"):
                    warnings.warn("Use basis set for selected backend.")
                    
                
                else:
                    continue
            #if inst.name not in basis_gates:
            #    continue
            qubit_indices = tuple(qubit.index for qubit in qubits)
            for item in qubit_indices:
                if item not in visited_qubits:
                    visited_qubits.append(item)

            start_time = max(curr_time[q] for q in qubit_indices)
            for q in qubit_indices:
                if start_time > curr_time[q]:
                    free_times[q].append([curr_time[q], start_time])
                    
            duration = cals.get(inst.name, qubit_indices, *inst.params).duration
            for q in qubit_indices:
                new_time = start_time + duration
                if output_in_dt:
                    new_time = int(new_time)
                curr_time[q] = new_time
        end_time = max(curr_time.values())
        for q, time in curr_time.items():
            if end_time > time:
                free_times[q].append([time, end_time])

        for item in free_times:
            for i in range(0,len(free_times[item])):
                val0 = end_time - free_times[item][i][0]
                val1 = end_time - free_times[item][i][1]

                free_times[item][i][0] = val1
                free_times[item][i][1] = val0

        for item in visited_qubits:
            if item not in free_times:
                free_times[item] = []
    
    else:
        #use time in seconds
        gate_times = backend.properties().gate_length
        
        free_times = defaultdict(list)
        curr_time = defaultdict(int)
        visited_qubits = []
        
        for i in range(len(circuit.data)-1,-1,-1):
            inst = circuit.data[i][0]
            qubits = circuit.data[i][1]
            clbits = circuit.data[i][1]

            if inst.name not in basis_gates:
            #commented out below. now will break with improper use

                if (inst.name != "measure") and (inst.name != "barrier"):
                    warnings.warn("Use basis set for selected backend.")
                
                else:
                    continue
            #old...adding check to make sure gate is on backend
            #if inst.name not in basis_gates:
            #    continue
                
            qubit_indices = tuple(qubit.index for qubit in qubits)
            for item in qubit_indices:
                if item not in visited_qubits:
                    visited_qubits.append(item)
            
            start_time = max(curr_time[q] for q in qubit_indices)
            for q in qubit_indices:
                if start_time > curr_time[q]:
                    free_times[q].append([curr_time[q], start_time])
            duration = gate_times(inst.name,qubit_indices)#cals.get(inst.name, qubit_indices, *inst.params).duration
            for q in qubit_indices:
                new_time = start_time + duration
                if output_in_dt:
                    new_time = int(new_time)
                curr_time[q] = new_time
        end_time = max(curr_time.values())
        for q, time in curr_time.items():
            if end_time > time:
                free_times[q].append([time, end_time])

        for item in free_times:
            for i in range(0,len(free_times[item])):
                val0 = end_time - free_times[item][i][0]
                val1 = end_time - free_times[item][i][1]

                free_times[item][i][0] = val1
                free_times[item][i][1] = val0

        for item in visited_qubits:
            if item not in free_times:
                free_times[item] = []
        
    return free_times


def return_fwd_rev(qc, backend=None, id_dt = 160):
    '''
    return fwd/rev circuit metrics 
        # for each slack window, we have 
            - 'fwd_circuit_all'
            - 'fwd_circuit'
            - 'slide_gates'
            - 'rev_circuit'
            - 'qubit_index'
            - 'slack_duration'
            - 'slack_id_buffer'
            - 'tuning_circ_depth'
    '''
    fwd_rev_circ_details = {}
    qc_copy = qc.copy()
    qc_adjusted, timing_for_counts, id_inst_label = return_adj_sched_circuit(qc_copy,backend, location = 1,return_verbose=True)
    
    timing_for_counts.reverse()
    id_inst_label.reverse()
    
    for i in range(len(timing_for_counts)):
        fwd_rev_circ_details[i] = {'qubit_index' : timing_for_counts[i][0],
                                   'slack_location': int(id_inst_label[i]),
                                   'slack_duration' : timing_for_counts[i][2]-timing_for_counts[i][1],
                                   'slack_id_buffer' : floor((timing_for_counts[i][2]-timing_for_counts[i][1])/id_dt)}
        
    for i in range(len(fwd_rev_circ_details)):
        
        #use betlow to include the classical registers of the original circuit
        #temp_circuit = QuantumCircuit(qc.qregs[0],qc.cregs[0])
        #use below to not include classical registers in fwd/rev circuits
        temp_circuit = QuantumCircuit(qc.qregs[0])
        for j in range(len(qc)):
            temp_circuit.append(qc[j][0],qc[j][1],qc[j][2])
            if j == fwd_rev_circ_details[i]['slack_location']:
                fwd_rev_circ_details[i]['fwd_circuit_all'] = temp_circuit.copy()
                inverse_circ = temp_circuit.inverse().copy()
                fwd_rev_circ_details[i]['rev_circuit'] = inverse_circ.copy()
                break
    
    
    keys_circs = fwd_rev_circ_details.keys()
    for i in keys_circs:
        slack_qubit = fwd_rev_circ_details[i]['qubit_index']
        temp_fwd_qc_holder = []
        temp_gate_holder = []
        gates_found = False
        for j in range(len(fwd_rev_circ_details[i]['fwd_circuit_all'])-1,-1,-1):
            local_qubits = []
            for qubit in fwd_rev_circ_details[i]['fwd_circuit_all'][j][1]:
                local_qubits.append(qubit.index)
               
            if gates_found ==False:
    
                if (fwd_rev_circ_details[i]['fwd_circuit_all'][j][0].name == 'cx' and slack_qubit in local_qubits):
                    gates_found = True
                    temp_fwd_qc_holder.append(fwd_rev_circ_details[i]['fwd_circuit_all'][j])
                elif slack_qubit not in local_qubits:
                    temp_fwd_qc_holder.append(fwd_rev_circ_details[i]['fwd_circuit_all'][j])
                    continue
                else:
                    #print(test_circs[i]['fwd_circuit'][j])
                    temp_gate_holder.append(fwd_rev_circ_details[i]['fwd_circuit_all'][j])
            else:
                temp_fwd_qc_holder.append(fwd_rev_circ_details[i]['fwd_circuit_all'][j])
                
        temp_gate_holder.reverse()
        
        #temp_slk_circuit = QuantumCircuit(fwd_rev_circ_details[i]['fwd_circuit_all'].qregs[0],fwd_rev_circ_details[i]['fwd_circuit_all'].cregs[0])
        temp_slk_circuit = QuantumCircuit(fwd_rev_circ_details[i]['fwd_circuit_all'].qregs[0])
        for j in range(len(temp_gate_holder)):
            temp_slk_circuit.append(temp_gate_holder[j][0],temp_gate_holder[j][1],temp_gate_holder[j][2])
        fwd_rev_circ_details[i]['slide_gates'] = temp_slk_circuit.copy()
        
        temp_fwd_qc_holder.reverse()
        #temp_fwd_circuit = QuantumCircuit(fwd_rev_circ_details[i]['fwd_circuit_all'].qregs[0],fwd_rev_circ_details[i]['fwd_circuit_all'].cregs[0])
        temp_fwd_circuit = QuantumCircuit(fwd_rev_circ_details[i]['fwd_circuit_all'].qregs[0])
        for j in range(len(temp_fwd_qc_holder)):
            temp_fwd_circuit.append(temp_fwd_qc_holder[j][0],temp_fwd_qc_holder[j][1],temp_fwd_qc_holder[j][2])
        fwd_rev_circ_details[i]['fwd_circuit'] = temp_fwd_circuit.copy()
        

    
    
    return fwd_rev_circ_details

def build_fwd_rev_circuits(fwd_circuit,slide_gates,rev_circuit,qubit_index,slack_id_buffer,id_interval=1):
    test_circuits_fwd_rev = []
    for i in range(0,slack_id_buffer+1,id_interval):
        qc = QuantumCircuit(fwd_circuit.qregs[0])
        qc = qc + fwd_circuit
                
        #delay before
        for j in range(0,slack_id_buffer-i):
            qc.id(qubit_index)
            
        qc = qc + slide_gates
        #delay after
        for j in range(0,int(i)):
            qc.id(qubit_index)
        qc = qc + rev_circuit
        qc.measure_active()   
        test_circuits_fwd_rev.append(qc)
    return test_circuits_fwd_rev


def return_all_test_circuits(fwd_rev_circuit_info, id_interval=1):
    all_fwd_rev = []
    
    for i in fwd_rev_circuit_info:
        all_fwd_rev = all_fwd_rev + build_fwd_rev_circuits(fwd_rev_circuit_info[i]['fwd_circuit'],fwd_rev_circuit_info[i]['slide_gates'],
                                                       fwd_rev_circuit_info[i]['rev_circuit'],fwd_rev_circuit_info[i]['qubit_index'],
                                                        fwd_rev_circuit_info[i]['slack_id_buffer'],id_interval=id_interval)
        
        fwd_rev_circuit_info[i]['tuning_circ_depth'] = find_cx_depth(all_fwd_rev[-1].copy())
        
    return all_fwd_rev


def find_cx_depth(circuit):
    dag_circuit = circuit_to_dag(circuit)
    cx_depth = dag_circuit.count_ops_longest_path()['cx']
    return cx_depth

def return_rescheduled_circuit(original_circuit,fwd_rev_circuit_info,max_locations):
    counter = 0
    rescheduled_circ = QuantumCircuit(original_circuit.qregs[0],original_circuit.cregs[0])
    slack_index_labels = []
    for i in range(len(fwd_rev_circuit_info)):
        slack_index_labels.append(fwd_rev_circuit_info[i]['slack_location'])

    labels_max_locations = []
    for i in range(len(slack_index_labels)):
        labels_max_locations.append([slack_index_labels[i],max_locations[i],fwd_rev_circuit_info[i]['qubit_index']])
    #print(slack_index_labels)
    #print(max_locations)    
    #print(labels_max_locations)
    
    labels_max_locations.sort()
    #print(labels_max_locations)
    
    for i in range(len(original_circuit)):
        if counter < len(slack_index_labels):
            if i == labels_max_locations[counter][0]:
                rescheduled_circ.append(original_circuit[i][0],original_circuit[i][1],original_circuit[i][2])
                for j in range(labels_max_locations[counter][1]):
                    rescheduled_circ.id(labels_max_locations[counter][2])
                counter = counter + 1
            else:
                rescheduled_circ.append(original_circuit[i][0],original_circuit[i][1],original_circuit[i][2])
            
        else:
            rescheduled_circ.append(original_circuit[i][0],original_circuit[i][1],original_circuit[i][2])
                
    
    return rescheduled_circ

def return_rescheduled_circuit_dd(original_circuit,backend,fwd_rev_circuit_info,max_locations):
    '''
    return circuits after SI tuning w/DD
    '''
    counter = 0
    rescheduled_circ = QuantumCircuit(original_circuit.qregs[0],original_circuit.cregs[0])
    ID_2_DD = 6 # hardcoded but may have to make this more flexible in the future...
    slack_index_labels = []
    for i in range(len(fwd_rev_circuit_info)):
        slack_index_labels.append(fwd_rev_circuit_info[i]['slack_location'])

    labels_max_locations = []
    for i in range(len(slack_index_labels)):
        labels_max_locations.append([slack_index_labels[i],max_locations[i],fwd_rev_circuit_info[i]['qubit_index'],fwd_rev_circuit_info[i]['slack_id_buffer']])
    #print(slack_index_labels)
    #print(max_locations)    
    #print(labels_max_locations)
    
    labels_max_locations.sort()
    #print(labels_max_locations)
    
    for i in range(len(original_circuit)):
        if counter < len(slack_index_labels):
            if i == labels_max_locations[counter][0]:
                rescheduled_circ.append(original_circuit[i][0],original_circuit[i][1],original_circuit[i][2])
                #for j in range(labels_max_locations[counter][1]):
                #    rescheduled_circ.id(labels_max_locations[counter][2])
                num_dd_seq = floor(labels_max_locations[counter][1]/ID_2_DD)
                num_extra_id = labels_max_locations[counter][1]%ID_2_DD
                for i in range(0,num_dd_seq):
                    rescheduled_circ.x(labels_max_locations[counter][2])
                    rescheduled_circ.y(labels_max_locations[counter][2])
                    rescheduled_circ.x(labels_max_locations[counter][2])
                    rescheduled_circ.y(labels_max_locations[counter][2])
                for i in range(0,num_extra_id):
                    rescheduled_circ.id(labels_max_locations[counter][2])
                
                counter = counter + 1
            else:
                rescheduled_circ.append(original_circuit[i][0],original_circuit[i][1],original_circuit[i][2])
            
        else:
            rescheduled_circ.append(original_circuit[i][0],original_circuit[i][1],original_circuit[i][2])
                
    #we have rescheduled circuit in rescheduled_circ. now we need to pad it!
    rescheduled_circ_transpiled = transpile(rescheduled_circ,backend,optimization_level=0)
    resched_circ_all_padded = return_adj_sched_circuit_dd(rescheduled_circ_transpiled,backend,location=0)
    resched_circ_all_padded_transpiled = transpile(resched_circ_all_padded,backend,optimization_level=0)
    return resched_circ_all_padded_transpiled


def pos_fwd_rev(counts,shots):
    correct_POS = []
    for thing in counts:
        reg_size = len(list(thing.keys())[0])
        correct_target = '0'*reg_size
        correct_POS.append((thing[correct_target])/shots)
    
    return correct_POS
        
def find_max_fwd_rev(fwd_rev_circuit_info,correct_POS,id_interval=1,produce_plot=False):
    max_locations = []

    counter = 0
    for i in range(len(fwd_rev_circuit_info)):
        slack_id_buffer = fwd_rev_circuit_info[i]['slack_id_buffer']
        
        
        x_vals = np.arange(slack_id_buffer+1,step=id_interval)
        y_vals = correct_POS[counter:counter + len(x_vals)]
    
        if produce_plot == True:
            print(f'total ID buffer in slack : {slack_id_buffer}')
            plt.ylabel('success probability - correct result')
            plt.xlabel('# ID gates b/w 1q gates and ALAP')
            plt.title(f'fwd/rev tuning experiments for slack window {i}')
            plt.figure(figsize=(13,10))
            plt.plot(x_vals,y_vals)
            
        max_locations.append(x_vals[y_vals.index(max(y_vals))])
        plt.show()
        counter = counter + len(x_vals)
    return max_locations

def calculate_hf(sim_counts,experiment_counts,):
    hf_values = []
    for i in range(0,len(experiment_counts)):
        hf_values.append(hellinger_fidelity(sim_counts,experiment_counts[i]))
    
    print(f'Average HF = {np.average(hf_values)}')
    return np.average(hf_values)

    
def calculate_pos(experiment_counts,correct_target,shots):
    pos_values = []
    for i in range(0,len(experiment_counts)):
        temp= 0
        for j in correct_target:
            temp = temp + experiment_counts[i][j]
        pos_values.append(temp/shots)
    
    print(f'Average POS = {np.average(pos_values)}')
    return np.average(pos_values)

def percent_change(v1,v2):
    change = (v2-v1)/v1
    print(f'Percent Change = {change}')
    return change





def create_circuit_w_delay_dd_single_seq(circuit,id_inst_label, id_inst_count, id_total_window,rounds=[0]*1000,seq='xy',spread=0,squeeze=0):
    '''
    pad with ID and single DD sequence in middle 
    '''
    
    qc = QuantumCircuit(circuit.num_qubits,circuit.num_clbits)
    
    ID_2_DD = 6 # hardcoded but may have to make this more flexible in the future...
    rounds_counter=0 
    #print("Here-2")
    for item in circuit:
        #print("Here-3")
        if item[0].label in id_inst_label:
            #print("Here-4")
            index_val = id_inst_label.index(item[0].label)
            
            qc.append(item[0],item[1],item[2])

            if floor(id_inst_count[index_val]/ID_2_DD) ==0: #GOKUL This means no rounds are possible and is a trivial case, so leaving this as is
                #print("Here-5")
                for i in range(0,id_inst_count[index_val]):
                    qc.id(item[1])
            else:
                #GOKUL First figure out if this many rounds is feasible, then use that with ration
                #print("Here-6")
                max_possible = floor(id_inst_count[index_val]/ID_2_DD)
                actual_rounds = min(max_possible,1+floor(rounds[rounds_counter]*max_possible)) #min(max_possible,rounds[rounds_counter]) 
                if(rounds[0]== -1):
                    actual_rounds=0
                ID_count_temp = max(0,floor(id_inst_count[index_val] - actual_rounds*ID_2_DD - squeeze*id_inst_count[index_val]))
                ID_count_side = floor(0.5*squeeze*id_inst_count[index_val])
                
                #First do the front
                for i in range(ID_count_side): #GOKUL
                    qc.id(item[1])                     
                
                if(spread==0): #This is the roginal chunk method
                    ID_block = floor((ID_count_temp/(actual_rounds+1)))
                    
                    for i in range(ID_block): #GOKUL - fffirst get one chunk in... rounds is 0, this is the total
                        qc.id(item[1]) 

                    #GOKUL - now get the other sets of (DD+block)
                    for i in range(actual_rounds):
                        if(seq=='xy'):
                            qc.x(item[1])
                            qc.y(item[1])
                            qc.x(item[1])
                            qc.y(item[1])
                        elif(seq=='xx'):
                            qc.x(item[1])
                            qc.x(item[1])
                            qc.x(item[1])
                            qc.x(item[1])       
                        elif(seq=='yy'):
                            qc.y(item[1])
                            qc.y(item[1])
                            qc.y(item[1])
                            qc.y(item[1])   
                        for i in range(ID_block):
                            qc.id(item[1])
                            
                if(spread==1): #This is the version Ali described
                    ID_block = floor((ID_count_temp/(4*actual_rounds+1)))
                    for i in range(ID_block): #GOKUL - fffirst get one chunk in... rounds is 0, this is the total
                        qc.id(item[1]) 

                    #GOKUL - now get the other sets of (DD+block)
                    for i in range(actual_rounds):
                        if(seq=='xy'):
                            qc.x(item[1])
                            for i in range(ID_block):
                                qc.id(item[1])                    
                            qc.y(item[1])
                            for i in range(ID_block):
                                qc.id(item[1])                                                
                            qc.x(item[1])
                            for i in range(ID_block):
                                qc.id(item[1])                    
                            qc.y(item[1])
                            for i in range(ID_block):
                                qc.id(item[1])                    
                        elif(seq=='xx'):
                            qc.x(item[1])
                            for i in range(ID_block):
                                qc.id(item[1])                    
                            qc.x(item[1])
                            for i in range(ID_block):
                                qc.id(item[1])                                                
                            qc.x(item[1])
                            for i in range(ID_block):
                                qc.id(item[1])                    
                            qc.x(item[1])
                            for i in range(ID_block):
                                qc.id(item[1])          
                        elif(seq=='yy'):
                            qc.y(item[1])
                            for i in range(ID_block):
                                qc.id(item[1])                    
                            qc.y(item[1])
                            for i in range(ID_block):
                                qc.id(item[1])                                                
                            qc.y(item[1])
                            for i in range(ID_block):
                                qc.id(item[1])                    
                            qc.y(item[1])
                            for i in range(ID_block):
                                qc.id(item[1])          

                #Last do the back
                for i in range(ID_count_side): #GOKUL 
                    qc.id(item[1])                         
                
            rounds_counter+=1
        else:
            #print("Here-7")
            #print(qc.qasm())
            qc.append(item[0],item[1],item[2])
    
    #we have rescheduled circuit in qc. now we need to pad it!
    
    #information needed to pad window
    qubits = [None]*len(id_inst_label)
    id_labels_padding = [None]*len(id_inst_label)
    id_padding_counts = []
    
    for i in range(len(id_inst_count)):
        id_padding_counts.append(id_total_window[i]-id_inst_count[i])
    
    #determine labels for where id padding should go
    for i in range(len(circuit)):
        #print(f'label:{circuit[i][0].label}')
        if circuit[i][0].label in id_inst_label:
            index_val = id_inst_label.index(circuit[i][0].label)
            #print(index_val)
            qubits[index_val] = circuit[i][1][0].index
            gate_found = False
            
            for j in range(i,-1,-1):
                if gate_found == True:
                    break
                
                local_qb = []
                for k in circuit[j][1]:
                    local_qb.append(k.index)
                
                #print(local_qb)

                if (circuit[j][0].name == 'cx') and qubits[index_val] in local_qb:
                    id_labels_padding[index_val] = circuit[j][0].label
                    gate_found = True
    
    #print(id_padding_counts)
    #print(qubits)
    #print(id_labels_padding)
    
    #create fully padded circuit 
    rounds_pad_counter=0 
    qc_padded = QuantumCircuit(circuit.num_qubits,circuit.num_clbits)
    for item in qc:
        if item[0].label in id_labels_padding:

            index_val = id_labels_padding.index(item[0].label)

            
            qc_padded.append(item[0],item[1],item[2])
            #for i in range(0,id_padding_counts[index_val]):
                #qc_padded.id(qubits[index_val])
            #num_dd_seq = floor(id_padding_counts[index_val]/ID_2_DD)
            #num_extra_id = id_padding_counts[index_val]%ID_2_DD
            #for i in range(0,num_dd_seq):
            #    qc_padded.x(qubits[index_val])
            #    qc_padded.y(qubits[index_val])
            #    qc_padded.x(qubits[index_val])
            #    qc_padded.y(qubits[index_val])
            #for i in range(0,num_extra_id):
            #    qc_padded.id(qubits[index_val])
            #
            if floor(id_padding_counts[index_val]/ID_2_DD) ==0:
                for i in range(0,id_padding_counts[index_val]):
                    qc_padded.id(qubits[index_val])
            else:

                #GOKUL First figure out if this many rounds is feasible, then use that with ration
                #print("Here-6")
                max_possible = floor(id_padding_counts[index_val]/ID_2_DD)
                actual_rounds = min(max_possible,1+floor(rounds[rounds_pad_counter]*max_possible))
                if(rounds[0]== -1):
                    actual_rounds=0
                #print(actual_rounds)
                ID_count_temp = max(0,floor(id_padding_counts[index_val] - actual_rounds*ID_2_DD - squeeze*id_padding_counts[index_val]))
                ID_count_side = floor(0.5*squeeze*id_padding_counts[index_val])
                #First do the front
                for i in range(ID_count_side): #GOKUL 
                    qc_padded.id(qubits[index_val])                    

                if(spread==0): #This is the roginal chunk method
                    ID_block = floor((ID_count_temp/(actual_rounds+1)))
                    
                    for i in range(ID_block): #GOKUL - fffirst get one chunk in... rounds is 0, this is the total
                        qc_padded.id(qubits[index_val]) 

                    #GOKUL - now get the other sets of (DD+block)
                    for i in range(actual_rounds):
                        if(seq=='xy'):
                            qc_padded.x(qubits[index_val])
                            qc_padded.y(qubits[index_val])
                            qc_padded.x(qubits[index_val])
                            qc_padded.y(qubits[index_val])
                        elif(seq=='xx'):
                            qc_padded.x(qubits[index_val])
                            qc_padded.x(qubits[index_val])
                            qc_padded.x(qubits[index_val])
                            qc_padded.x(qubits[index_val])       
                        elif(seq=='yy'):
                            qc_padded.y(qubits[index_val])
                            qc_padded.y(qubits[index_val])
                            qc_padded.y(qubits[index_val])
                            qc_padded.y(qubits[index_val])   
                        for i in range(ID_block):
                            qc_padded.id(qubits[index_val])
                            
                if(spread==1): #This is the version Ali described
                    ID_block = floor((ID_count_temp/(4*actual_rounds+1)))
                    for i in range(ID_block): #GOKUL - fffirst get one chunk in... rounds is 0, this is the total
                        qc_padded.id(qubits[index_val]) 

                    #GOKUL - now get the other sets of (DD+block)
                    for i in range(actual_rounds):
                        if(seq=='xy'):
                            qc_padded.x(qubits[index_val])
                            for i in range(ID_block):
                                qc_padded.id(qubits[index_val])                    
                            qc_padded.y(qubits[index_val])
                            for i in range(ID_block):
                                qc_padded.id(qubits[index_val])                                                
                            qc_padded.x(qubits[index_val])
                            for i in range(ID_block):
                                qc_padded.id(qubits[index_val])                    
                            qc_padded.y(qubits[index_val])
                            for i in range(ID_block):
                                qc_padded.id(qubits[index_val])                    
                        elif(seq=='xx'):
                            qc_padded.x(qubits[index_val])
                            for i in range(ID_block):
                                qc_padded.id(qubits[index_val])                    
                            qc_padded.x(qubits[index_val])
                            for i in range(ID_block):
                                qc_padded.id(qubits[index_val])                                                
                            qc_padded.x(qubits[index_val])
                            for i in range(ID_block):
                                qc_padded.id(qubits[index_val])                    
                            qc_padded.x(qubits[index_val])
                            for i in range(ID_block):
                                qc_padded.id(qubits[index_val])          
                        elif(seq=='yy'):
                            qc_padded.y(qubits[index_val])
                            for i in range(ID_block):
                                qc_padded.id(qubits[index_val])                    
                            qc_padded.y(qubits[index_val])
                            for i in range(ID_block):
                                qc_padded.id(qubits[index_val])                                                
                            qc_padded.y(qubits[index_val])
                            for i in range(ID_block):
                                qc_padded.id(qubits[index_val])                    
                            qc_padded.y(qubits[index_val])
                            for i in range(ID_block):
                                qc_padded.id(qubits[index_val])        

                #Last do the back
                for i in range(ID_count_side): #GOKUL 
                    qc_padded.id(qubits[index_val])                                    
                    
            rounds_pad_counter+=1
                    
                    
                    
        else:
            qc_padded.append(item[0],item[1],item[2])
    
    
    #quick workaround to remove label
    for item in qc_padded:
        item[0].label = None
    
    return qc_padded


    


def create_circuit_w_delay_dd_full_padding(circuit,id_inst_label, id_inst_count, id_total_window):
    '''
    full window padding with XYXY (++ some extra ID gates, if necessary)
    '''
    
    qc = QuantumCircuit(circuit.num_qubits,circuit.num_clbits)
    #print("Here-10")
    ID_2_DD = 6 # hardcoded but may have to make this more flexible in the future...
    
    for item in circuit:
        if item[0].label in id_inst_label:

            index_val = id_inst_label.index(item[0].label)
            
            #debugging..can comment out later
            
            #print_gate_info(item)
            #print(f'id count:{id_inst_count[index_val]}')
            #print(f'label: {id_inst_label[index_val]}')
            
            qc.append(item[0],item[1],item[2])
            #for i in range(0,id_inst_count[index_val]):
            #    qc.id(item[1])
            num_dd_seq = floor(id_inst_count[index_val]/ID_2_DD)
            num_extra_id = id_inst_count[index_val]%ID_2_DD
            for i in range(0,num_dd_seq):
                qc.x(item[1])
                qc.y(item[1])
                qc.x(item[1])
                qc.y(item[1])
            for i in range(0,num_extra_id):
                qc.id(item[1])
                
        else:
            #print(qc.qasm())
            qc.append(item[0],item[1],item[2])
    
    #we have rescheduled circuit in qc. now we need to pad it!
    
    #information needed to pad window
    qubits = [None]*len(id_inst_label)
    id_labels_padding = [None]*len(id_inst_label)
    id_padding_counts = []
    
    for i in range(len(id_inst_count)):
        id_padding_counts.append(id_total_window[i]-id_inst_count[i])
    
    #determine labels for where id padding should go
    for i in range(len(circuit)):
        #print(f'label:{circuit[i][0].label}')
        if circuit[i][0].label in id_inst_label:
            index_val = id_inst_label.index(circuit[i][0].label)
            #print(index_val)
            qubits[index_val] = circuit[i][1][0].index
            gate_found = False
            
            for j in range(i,-1,-1):
                if gate_found == True:
                    break
                
                local_qb = []
                for k in circuit[j][1]:
                    local_qb.append(k.index)
                
                #print(local_qb)

                if (circuit[j][0].name == 'cx') and qubits[index_val] in local_qb:
                    id_labels_padding[index_val] = circuit[j][0].label
                    gate_found = True
    
    #print(id_padding_counts)
    #print(qubits)
    #print(id_labels_padding)
    
    #create fully padded circuit 
    qc_padded = QuantumCircuit(circuit.num_qubits,circuit.num_clbits)
    for item in qc:
        if item[0].label in id_labels_padding:

            index_val = id_labels_padding.index(item[0].label)

            
            qc_padded.append(item[0],item[1],item[2])
            #for i in range(0,id_padding_counts[index_val]):
                #qc_padded.id(qubits[index_val])
            num_dd_seq = floor(id_padding_counts[index_val]/ID_2_DD)
            num_extra_id = id_padding_counts[index_val]%ID_2_DD
            for i in range(0,num_dd_seq):
                qc_padded.x(qubits[index_val])
                qc_padded.y(qubits[index_val])
                qc_padded.x(qubits[index_val])
                qc_padded.y(qubits[index_val])
            for i in range(0,num_extra_id):
                qc_padded.id(qubits[index_val])
                
        else:
            qc_padded.append(item[0],item[1],item[2])
    
    
    #quick workaround to remove label
    for item in qc_padded:
        item[0].label = None
    
    return qc_padded


    
    
    
def return_adj_sched_circuit_dd(qc, backend=None, output_in_dt=True, location = 0.5, use_dd_padding=True, dd_style='full', fixed_placement = True, use_delay = False, return_verbose =False,DELAY_VECT = [0]*1000,rounds=[0]*1000,seq='xy',spread=0,squeeze=0):
    '''
    move single qubit gates in circuit from alap to adjusted schedule
    returns - 
        qc_new - the adjusted quantum circuit w/ DD
        
    warning - unexpected results if basis gates are not used for a backend!
    
    dd_style - can be 'full' or 'single' 
    '''


    if backend is None:
        warnings.warn("Backend needed to produce timing information.")
    
    basis_gates = backend.configuration().basis_gates
    dt = 1
    
    circuit = deepcopy(qc)
    
    for i in range(0,len(circuit)):
        try:
            circuit[i][0].label = str(i)
        except:
            continue
            
    if output_in_dt:
        #use time in dt
        dt = backend.configuration().dt
        cals = backend.defaults().instruction_schedule_map
        
    else:
        #use time in seconds
        gate_times = backend.properties().gate_length

    free_times = defaultdict(list)
    curr_time = defaultdict(int)
    visited_qubits = []
        
    #holds visited qubits and associated labels...will be used to identify where to put id gates
    qubit_tracker = []
    id_inst_label = []
    
    timing_for_counts = []
    
    
        
    for i in range(len(circuit.data)-1,-1,-1):
        inst = circuit.data[i][0]
        label = inst.label
        qubits = circuit.data[i][1]
        clbits = circuit.data[i][1]
        if inst.name not in basis_gates:
            #commented out below. now will break with improper use
            #continue
            if (inst.name != "measure") and (inst.name != "barrier"):
                warnings.warn("Use basis set for selected backend.")
                
            else:
                continue
        qubit_indices = tuple(qubit.index for qubit in qubits)
        for item in qubit_indices:
            if item not in visited_qubits:
                visited_qubits.append(item)
        start_time = max(curr_time[q] for q in qubit_indices)
        for q in qubit_indices:
            if start_time > curr_time[q]:
                free_times[q].append([curr_time[q], start_time])
               
                for j in range(len(qubit_tracker)-1,-1,-1):
                    if qubit_tracker[j][0] == q :
                    #if qubit_tracker[j][0] == q:
                        #added below
                        if qubit_tracker[j][2] =='cx':
                            break
                        
                        temp_label = qubit_tracker[j][1]
                        qubit_number = qubit_tracker[j][0]
                        end_group_found = False
                        
                        for k in range(len(circuit.data)):
                            if circuit.data[k][0].label == temp_label:
                                temp_circ_index = k
                                local_qubits = tuple(qubit.index for qubit in circuit.data[temp_circ_index][1])
                                stop_gates = ['cx','barrier','measure']

                                
                                while ((circuit.data[temp_circ_index][0].name not in stop_gates) and (qubit_number in local_qubits)) :
                                    temp_circ_index = temp_circ_index + 1
                                    local_qubits = tuple(qubit.index for qubit in circuit.data[temp_circ_index][1])
                                    
                                
                                end_group_found = True
                                temp_label = circuit.data[temp_circ_index-1][0].label
                                
                                
                            if end_group_found ==True:
                                break
                         
                        id_inst_label.append(temp_label)
                        #old
                        #id_inst_label.append(qubit_tracker[j][1])
                        
                        
                        
                        
                        timing_for_counts.append((q,curr_time[q],start_time))
                        break
        
        
        if output_in_dt:
            duration = cals.get(inst.name, qubit_indices, *inst.params).duration
        else:
            duration = gate_times(inst.name,qubit_indices)
        for q in qubit_indices:
            new_time = start_time + duration
            if output_in_dt:
                new_time = int(new_time)
            curr_time[q] = new_time
            
            #edited this
            #add qubits to tracker if single qubit gate
            #if inst.name != 'cx':
            #    qubit_tracker.append([q,label])
            qubit_tracker.append([q,label,inst.name])
            
    end_time = max(curr_time.values())
    
    
    if use_delay == False:
        if fixed_placement == True:
            #if all gates moved to same point
            delay_vect = delay_vect_static_val(len(id_inst_label),location)

            id_inst_count = [] 
            id_total_window = []
            for i in range(0,len(timing_for_counts)):
                id_inst_count.append(count_id_needed(backend,
                                                     timing_for_counts[i][0],
                                                     timing_for_counts[i][1],
                                                     timing_for_counts[i][2],
                                                     location=delay_vect[i]
                                                     ,output_in_dt=output_in_dt))
                id_total_window.append(count_id_for_window(backend,
                                                     timing_for_counts[i][0],
                                                     timing_for_counts[i][1],
                                                     timing_for_counts[i][2],
                                                     output_in_dt=output_in_dt))
        else:
            #optimized_vect_search()
            #GOKUL = Heavily Modified
            delay_vect = DELAY_VECT

            id_inst_count = [] 
            id_total_window = []
            other_count = []
            for i in range(0,len(timing_for_counts)):
                id_inst_count.append(count_id_needed(backend,
                                                     timing_for_counts[i][0],
                                                     timing_for_counts[i][1],
                                                     timing_for_counts[i][2],
                                                     location=delay_vect[i]
                                                     ,output_in_dt=output_in_dt)) #GOKUL 
                
                id_total_window.append(count_id_for_window(backend,
                                                     timing_for_counts[i][0],
                                                     timing_for_counts[i][1],
                                                     timing_for_counts[i][2],
                                                     output_in_dt=output_in_dt))
                
                other_count.append(timing_for_counts[i])
            #return 0


        
        if use_dd_padding == False:
            qc_new = create_circuit_w_delay_id(circuit,id_inst_label, id_inst_count)
        if use_dd_padding == True:
            if dd_style == 'full':
                qc_new = create_circuit_w_delay_dd_full_padding(circuit,id_inst_label, id_inst_count,id_total_window)
                
            if dd_style == 'single':
                #print("Here")
                qc_new = create_circuit_w_delay_dd_single_seq(circuit,id_inst_label, id_inst_count,id_total_window,rounds=rounds,seq=seq,spread=spread,squeeze=squeeze)
    
    elif use_delay == True:
        if fixed_placement == True:
            #if all gates moved to same point
            delay_vect = delay_vect_static_val(len(id_inst_label),location)
            
            delay_gate_vals = []
            for i in range(0,len(timing_for_counts)):
                delay_gate_vals.append((timing_for_counts[i][2]-timing_for_counts[i][1])*delay_vect[i])
                
            qc_new = create_circuit_w_delay_gate(circuit,id_inst_label, delay_gate_vals,output_in_dt)
            
        else:
            optimized_vect_search()
            return 0

        
        
    if return_verbose == True:
        
        return qc_new, timing_for_counts, id_inst_label
    else:
        return qc_new, timing_for_counts