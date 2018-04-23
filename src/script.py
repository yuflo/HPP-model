import os
import sys
import threading
from time import time as tct
from gc import collect
import numpy as np
from scipy.sparse import csr_matrix

ext_csrm = '.csrm'
ext_map = '.map'
ext_hin = '.hin'
rc_sep = '-'
lock = threading.Lock()

''' transform .hin file to standard data'''
def hin2data(num_threads, ddr, hin_file):

    num_lines = 0

    fi = open(hin_file)
    line = fi.readline()
    while line != '':
        line = fi.readline()
        num_lines += 1

    vtypes = set()
    binames = set()
    tail = 1
    d = num_lines / num_threads

    print('Seperating hin_file...')
    # work 0
    for i in range(num_threads):
        mq = Worker(i+1, hin_file, ddr, vtypes, binames)
        mq.head = tail
        tail = num_lines+1 if i == num_threads-1 else tail+d
        mq.tail = tail
        mq.work = 0
        mq.setDaemon(True)
        mq.start()
        mq.join()

    print('Generating intermediate .csrm file and standard .map file...')
    # work 1 & 2
    mqs = []
    for i, biname in enumerate(binames):
        mq = Worker(i+1, hin_file, ddr, vtypes, binames)
        mq.work = 1
        mq.work_params['biname'] = biname
        mqs.append(mq)
    for i, vtype in enumerate(vtypes):
        mq = Worker(i+1, hin_file, ddr, vtypes, binames)
        mq.work = 2
        mq.work_params['vtype'] = vtype
        mqs.append(mq)
    for mq in mqs:
        mq.work_params['cut_id_set'] = list(range(1,num_threads+1))
        mq.setDaemon(True)
        mq.start()
        mq.join()

    print('Generating standard .csrm file...')
    #work 3
    for i, biname in enumerate(binames):
        mq = Worker(i+1, hin_file, ddr, vtypes, binames)
        mq.work = 3
        mq.work_params['biname'] = biname
        mq.setDaemon(True)
        mq.start()
        mq.join()

    print('Generating meta-path file...')
    with open(ddr+'/meta-path.txt', 'w') as fo:
        for biname in binames:
            v, c = biname.split('-')
            fo.write(biname+'\n')
            fo.write(c+'-'+v+'\n')
        
    print('Finish.')

class Worker(threading.Thread):
    def __init__(self, work_id, hin_file, ddr, vtypes, binames):
        threading.Thread.__init__(self)
        self.work_id = work_id
        self.hin_file = hin_file
        self.ddr = ddr
        self.head = 0
        self.tail = 0
        self.work = 0
        self.vtypes = vtypes
        self.binames = binames
        self.work_params = dict()

    def cut_map(self, hin_file, ddr, head, tail, vtypes):

        fi = open(hin_file)
        fos = dict()
        #print('worker %d' % (self.work_id))

        i = 0
        line = fi.readline()
        while line != '':
            i += 1
            if i < head: # [head, tail)
                line = fi.readline()
            elif i < tail:
                v, c, _ = line.strip().split(' ')
                vtype, vname = v.split('#')
                ctype, cname = c.split('#')

                try:
                    fo_v = fos[vtype]
                except:
                    fo_v = open(ddr+'/'+vtype+str(self.work_id)+'.map_tmp', 'w')
                    fos[vtype] = fo_v
                fo_v.write(vname+'\n')
                line = fi.readline()

                if vtype == ctype:
                    continue

                try:
                    fo_c = fos[ctype]
                except:
                    fo_c = open(ddr+'/'+ctype+str(self.work_id)+'.map_tmp', 'w')
                    fos[ctype] = fo_c
                fo_c.write(cname+'\n')

            else:
                break

        for vtype, fo in fos.iteritems():
            fo.close()

        for vtype in fos.iterkeys():
            lock.acquire()
            try:
                vtypes.add(vtype)
            finally:
                lock.release()

    def cut_csrm(self, hin_file, ddr, head, tail, binames):

        fi = open(hin_file)
        fos = dict()
        #print('worker %d' % (self.work_id))

        i = 0
        line = fi.readline()
        while line != '':
            i += 1
            if i < head: # [head, tail)
                line = fi.readline()
            elif i < tail:

                v, c, w= line.strip().split(' ')
                vtype, vname = v.split('#')
                ctype, cname = c.split('#')
                biname = vtype+rc_sep+ctype

                try:
                    fo_v = fos[biname]
                except:
                    fo_v = open(ddr+'/'+biname+str(self.work_id)+'.csrm_tmp', 'w')
                    fos[biname] = fo_v

                fo_v.write(vname+' '+cname+' '+w+'\n')
                line = fi.readline()
            else:
                break

        for biname, fo in fos.iteritems():
            fo.close()

        for biname in fos.iterkeys():
            lock.acquire()
            try:
                binames.add(biname)
            finally:
                lock.release()

    def merge_csrm(self, ddr, biname, cut_id_set):
        fo = open(ddr+'/'+biname+'.csrm_tmp', 'w')
        for cut_id in cut_id_set:
            filename = ddr+'/'+biname+str(cut_id)+'.csrm_tmp'
            if not os.path.exists(filename):
                continue

            fi = open(filename)
            line = fi.readline()
            while line != '':
                fo.write(line)
                line = fi.readline()
            os.remove(filename)

    def merge_map(self, ddr, vtype, cut_id_set):

        fo = open(ddr+'/'+vtype+'.map', 'w')
        vset = set()

        for cut_id in cut_id_set:
            filename = ddr+'/'+vtype+str(cut_id)+'.map_tmp'
            if not os.path.exists(filename):
                continue

            fi = open(filename)
            line = fi.readline()
            while line != '':
                vset.add(line.strip())
                line = fi.readline()
            os.remove(filename)

        for i, v in enumerate(vset):
            fo.write(str(i)+' '+v+'\n')
        fo.close()

        vset = None
        collect()

    def tocsrm(self, ddr, biname):

        ## load in map
        name2id_v = dict()
        name2id_c = dict()

        v, c = biname.split('-')

        fi = open(ddr+'/'+v+'.map')
        line = fi.readline()
        while line != '':
            id_, name = line.strip().split(' ')
            name2id_v[name] = id_
            line = fi.readline()

        fi = open(ddr+'/'+c+'.map')
        line = fi.readline()
        while line != '':
            id_, name = line.strip().split(' ')
            name2id_c[name] = id_
            line = fi.readline()

        vn = len(name2id_v)
        cn = len(name2id_c)
        nnz = 0
        degs_v = [0] * vn
        degs_c = [0] * cn
        edges = dict()

        # merging edges
        fi = open(ddr+'/'+biname+'.csrm_tmp')
        line = fi.readline()
        while line != '':
            line = line.strip()
            v, c, w = line.split(' ')
            key = v+'-'+c
            try:
                edges[key] += float(w)
            except:
                edges[key] = float(w)
                degs_v[int(name2id_v[v])] += 1
                degs_c[int(name2id_c[c])] += 1
                nnz += 1
            line = fi.readline()

        # conduct csr_mat data structure and output
        indptr_v = [0] * (vn + 1)
        for i, d in enumerate(degs_v):
            indptr_v[i+1] = indptr_v[i] + d
        indptr_c = [0] * (cn + 1)
        for i, d in enumerate(degs_c):
            indptr_c[i+1] = indptr_c[i] + d
            
        indices_v = [0] * nnz
        indices_c = [0] * nnz
        data_v = [0.] * nnz
        data_c = [0.] * nnz
        for edge, w in edges.iteritems():
            v, c = edge.split('-')
            vi, ci = int(name2id_v[v]), int(name2id_c[c])

            idx_v = indptr_v[vi] + degs_v[vi] - 1
            indices_v[idx_v] = ci
            data_v[idx_v] = w
            degs_v[vi] -= 1

            idx_c = indptr_c[ci] + degs_c[ci] - 1
            indices_c[idx_c] = vi
            data_c[idx_c] = w
            degs_c[ci] -= 1

        os.remove(ddr+'/'+biname+'.csrm_tmp')
        edges = None
        collect=()

        v, c = biname.split('-')
        self.outcsrm(vn, cn, nnz, indptr_v, indices_v, data_v, ddr+'/'+biname+'.csrm')
        self.outcsrm(cn, vn, nnz, indptr_c, indices_c, data_c, ddr+'/'+c+'-'+v+'.csrm')

    def outcsrm(self, rn, cn, nnz, indptr, indices, data, outfile):
        with open(outfile, 'w') as fo:
            fo.write(str(rn)+' '+str(cn)+' '+str(nnz)+'\n')
            for d in indptr:
                fo.write(str(d)+' ')
            fo.write('\n')
            for d in indices:
                fo.write(str(d)+' ')
            fo.write('\n')
            for d in data:
                fo.write(str(d)+' ')
            fo.write('\n')


    def run(self):
        if self.work == 0:
            self.cut_csrm(self.hin_file, self.ddr, self.head, self.tail, self.binames)
            self.cut_map(self.hin_file, self.ddr, self.head, self.tail, self.vtypes)
        elif self.work == 1:
            self.merge_csrm(self.ddr, self.work_params['biname'], self.work_params['cut_id_set'])
        elif self.work == 2:
            self.merge_map(self.ddr, self.work_params['vtype'], self.work_params['cut_id_set'])
        elif self.work == 3:
            self.tocsrm(self.ddr, self.work_params['biname'])

''' transform .npy file to .hin file'''
def npy2hin(bimap, ddr=os.curdir, net_name='unknown'):

    if not os.path.exists(ddr):
        os.mkdir(ddr)

    print('Transform .npy files to .hin file...')

    with open(ddr+'/'+net_name+ext_hin, 'w') as fo:
        for biname, smat in bimap.items():
            vname, cname = biname.split(rc_sep)
            indptr, indices, data = smat.indptr, smat.indices, smat.data 
            rn, cn = smat.shape
            nnz = smat.nnz
            for i in range(rn):
                s, t = indptr[i], indptr[i+1]
                for j in range(s, t):
                    fo.write(vname+'#'+str(i)+' '+cname+'#'+str(indices[j])+' '+str(data[j])+'\n')

    print('Finish.')

''' transform .npy file to standard data files (.csrm + .map + meta-pair.txt)'''
def npy2data(bimap, ddr=os.curdir):

    if not os.path.exists(ddr):
        os.mkdir(ddr)

    map_keys = set()

    print('Transform .npy files to standard data files...')

    for biname, smat in bimap.iteritems():
        vname, cname = biname.split(rc_sep)
        fname_csrm = ddr+'/'+biname+ext_csrm
        fname_csrm_r = ddr+'/'+cname+rc_sep+vname+ext_csrm

        if vname not in map_keys:
            map_keys.add(vname)
            fname_vmap = ddr+'/'+vname+ext_map
        if cname not in map_keys:
            map_keys.add(cname)
            fname_cmap = ddr+'/'+cname+ext_map

        smat_T = csr_matrix(smat.T)
        outsmat(smat, fname_csrm, fname_vmap, fname_cmap)
        outsmat(smat_T, fname_csrm_r)

    with open(ddr+'/meta-pair.txt', 'w') as fo:
        for biname in bimap.keys():
            vname, cname = biname.split(rc_sep)
            fo.write(biname+'\n')
            fo.write(cname+rc_sep+vname+'\n')

    print('Finish.')

def outsmat(smat, fname_csrm=None, fname_vmap=None, fname_cmap=None):

    indptr, indices, data = smat.indptr, smat.indices, smat.data 
    rn, cn = smat.shape
    nnz = smat.nnz

    if fname_csrm is None:
        return

    with open(fname_csrm, 'w') as fo_vc:
        fo_vc.write(str(rn)+' '+str(cn)+' '+str(nnz)+'\n')
        for d in indptr:
            fo_vc.write(str(d)+' ')
        fo_vc.write('\n')
        for d in indices:
            fo_vc.write(str(d)+' ')
        fo_vc.write('\n')
        for d in data:
            fo_vc.write(str(d)+' ')
        fo_vc.write('\n')

    if fname_vmap is not None:
        with open(fname_vmap, 'w') as fo_v:
            for i in range(rn):
                fo_v.write(str(i)+' '+str(i)+'\n')

    if fname_cmap is not None:
        with open(fname_cmap, 'w') as fo_c:
            for i in range(cn):
                fo_c.write(str(i)+' '+str(i)+'\n')
        

if __name__ == '__main__':

    if len(sys.argv) == 1:
        print('Notice that, npy file here stores sparse matrix, particullarly, csr_matrix format. Please refer scipy.org for more details.')
        print('Options:')
        print('\t-operation <transform operation>')
        print('\t\tThree types of operation, use 1/2/3 to indicate: npy2hin (1), npy2data (2), hin2data(3)')
        print('\t-input <.hin_file / name-npy pairs>')
        print('\t\t1. hin_file is the standard file path format\n\t\t2. name-npy use key-file_path format, e.g.p-a:ac.npy,p-c:pc.npy')
        print('\t-output <directory path>')
        print('\t\tDirectory path for outputing data files')
        print('\t-net_name <network name>')
        print('\t\tValid only for npy2hin operation')
        print('\t-threads <int>')
        print('\t\tValid only for hin2data operation')
    else:
        params = {'-operation':'3', '-input':'', '-output':'', '-net_name':'unknown', '-threads':'5'}
        param_key = ''

        for argv in sys.argv[1:]:
            if param_key != '':
                params[param_key] = argv
                param_key = ''
                continue
            param_key = argv

        operation = int(params['-operation'])
        input_ = params['-input']
        output = params['-output']
        net_name = params['-net_name']
        threads = int(params['-threads'])

        if operation == 3:
            hin2data(threads, output, input_)
        else:
            bimap = dict()
            pairs = input_.split(',')
            for pair in pairs:
                key, file_path = pair.split(':')
                bimap[key] = np.load(file_path)[()]
            if operation == 1:
                npy2hin(bimap, output, net_name)
            elif operation == 2:
                npy2data(bimap, output)
