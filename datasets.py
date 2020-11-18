from pathlib import Path

# Base paths to datasets and whatnot, so that there's only one place to change these if moving between computers or whatever

def get_kitti_mots_base():
    p = Path('/media/hdd/KITTI_MOTS')
    if p.is_dir():
        return p
    else:
        raise ValueError(f"Could not find {p}. Mounting error? Incorrect path?")

def get_classes(dataset):
    if dataset.startswith('KITTI'):
        return {2: 'person', 1: 'car', 3: 'bike'}
    else:
        return {2: 'person'}
    
def get_seqs(s):
    if s.dataset == 'MOTS':
        train_seqs = ['MOTS20-02', 'MOTS20-05', 'MOTS20-09', 'MOTS20-11']
        test_seqs = ['MOTS20-01', 'MOTS20-06', 'MOTS20-07', 'MOTS20-12']
        first_seqs = ['MOTS20-02']
        lowres_seqs = ['MOTS20-02', 'MOTS20-09', 'MOTS20-11']
        nm_train_seqs = ['MOTS20-02', 'MOTS20-05', 'MOTS20-11']
        nm_val_seqs = ['MOTS20-09']
        
        if s.set == 'train':
            seqs = train_seqs
            s.set_folder = 'train'
        elif s.set == 'test':
            seqs = test_seqs
            s.set_folder = 'test'
        elif s.set == 'first':
            seqs = first_seqs
            s.set_folder = 'train'
        elif s.set == 'lowres':
            seqs = lowres_seqs
            s.set_folder = 'train'
        elif s.set == 'neldermead_train':
            seqs = nm_train_seqs
            s.set_folder = 'train'
        elif s.set == 'neldermead_val':
            seqs = nm_val_seqs
            s.set_folder = 'train'
        else:
            raise ValueError(f"Incorrect set {s.set}")
        
        return seqs
    elif s.dataset == 'KITTI': #TODO
        if s.set == 'train':
            s.set_folder = 'training'
            seqs = extract_seqs(Path('KITTI') / 'train.seqmap')
        elif s.set == 'val':
            s.set_folder = 'training'
            seqs = extract_seqs(Path('KITTI') / 'val.seqmap')
        elif s.set == 'test':
            s.set_folder = 'testing'
            seqs = extract_seqs(Path('KITTI') / 'test.seqmap')
        
        return seqs
    
def extract_seqs(path):
    lines = [x for x in path.read_text().split('\n') if x]
    return [x.split(' ')[0] for x in lines]
    
