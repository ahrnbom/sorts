from datetime import datetime, timedelta

class Timing:
    def __init__(self):
        self.totals = dict()
        self.currents = dict()
        self.running = set()
        
    def start(self, key):
        if len(self.running):
            raise ValueError(f"Overlap! {key} {self.running}")
        
        if key in self.running:
            raise ValueError(f"Already running {key}")
            
        if not key in self.totals:
            self.totals[key] = timedelta()

        self.currents[key] = datetime.now()
        self.running.add(key)
    
    def stop(self, key, whatever=False):
        if not key in self.running:
            if whatever:
                return
            else:
                raise ValueError(f"{key} was not running!")
    
        dt = datetime.now() - self.currents[key]
        self.currents[key] = None
        
        self.totals[key] += dt
        
        self.running.remove(key)
    
    def print_totals(self, to_file=None):
        if self.running:
            raise ValueError(f"Not empty when printing! Still measuring {self.running}")
            
        for key in self.totals.keys():
            print(key, self.totals[key], file=to_file)
    
    def total_time(self):
        total = timedelta()
        for key in self.totals.keys():
            total += self.totals[key]
        return total
        
    def __add__(self, t):
        new = Timing()
        for key in self.totals:
            if key in t.totals:
                new.totals[key] = self.totals[key] + t.totals[key]
            else:
                new.totals[key] = self.totals[key]
        for key in t.totals:
            if not key in new.totals:
                if key in self.totals:
                    raise ValueError("Where is your God now?")
                new.totals[key] = t.totals[key]
        return new
    
    def __repr__(self):
        return self.print_totals()
    
    def __str__(self):
        return self.print_totals()

