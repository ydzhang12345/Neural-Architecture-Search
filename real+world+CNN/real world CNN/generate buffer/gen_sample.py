'''
import itertools
filter_val = [10, 50, 100, 200]
stride_val = [1,2]
kernel_val = [3,5]
filter_space = [v for v in itertools.product(filter_val, repeat=3)]
stride_space = [v for v in itertools.product(stride_val, repeat=3)]
kernel_space = [v for v in itertools.product(kernel_val, repeat=3)]

f_val = [50, 100, 200]
s_val = [1,2]
k_val = [3]
f_space = [v for v in itertools.product(f_val, repeat=3)]
s_space = [v for v in itertools.product(s_val, repeat=3)]
k_space = [v for v in itertools.product(k_val, repeat=3)]

main_state = []
with open('Main_case.txt', 'w') as out:
    j = 0
    for k in k_space:
       for s in s_space:
            for f in f_space:
                main_state.append([f[0],k[0],s[0],
                              f[1],k[1],s[1],
                              f[2],k[2],s[2]])
                sta = [f[0],k[0],s[0],
                              f[1],k[1],s[1],
                              f[2],k[2],s[2]]
                j += 1
                for st in sta:
                    out.write(str(st)+' ')
                out.write('\n')

print j

print filter_space,len(filter_space)
print stride_space,len(stride_space)
print kernel_space,len(kernel_space)
with open('All_case.txt', 'w') as outfile:
    k = 0
    for kernel in kernel_space:
       for stride in stride_space:
            for filters in filter_space:
                state = [filters[0],kernel[0],stride[0],
                              filters[1],kernel[1],stride[1],
                              filters[2],kernel[2],stride[2]]
                
                if state not in main_state:
                    k +=1
                    for stat in state:
                        outfile.write(str(stat)+' ')
                    outfile.write('\n')
                else:
                    pass

print k
'''

state = []
with open('All_case.txt', 'r') as fin:
    for line in fin:
        line = line.strip('\n')
        line = line.strip()
        line = line.split(' ')
        

        
        print line
        s = []
        for l in line:
            s.append(int(l))
        state.append(s)
R2 = state[500:1000]
print R2
print len(R2)

















