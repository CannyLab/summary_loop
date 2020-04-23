from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
import json, os

def geom(data, alpha=0.995):
    gdata = [data[0]]
    for d in data[1:]:
        gdata.append(gdata[-1]*alpha + (1-alpha)*d)
    return gdata

def plot(files=[], folder=None, filt='', geo=0, xaxis='steps', maxy={}):
	# filt: a filter substring. Only keys with this substring will be plotted. For example 'V' to plot only validation values
	# geo: a float between (0,1.0) that is used for smoothing the series. Values closer to 1.0 lead to more smoothing. 0.99 is a lot of smoothing
	# xaxis: what to base the x-axis on. `steps` will just be a range(len(values)). `datetime` will show absolute times.

	if folder is not None:
		files = [os.path.join(folder, fn) for fn in os.listdir(folder) if os.path.isfile(os.path.join(folder, fn))]

	bad_keys = ['datetime']
	full_summary = {}
	for file in files:
		full_summ = []
		with open(file, "r") as f:
			for line in f:
				if len(line) == 0: continue
				try: obj = json.loads(line)
				except:
					obj = {}
				if 'datetime' in obj:
					obj['datetime'] = datetime.strptime(obj['datetime'][:19], "%Y-%m-%d %H:%M:%S")
				full_summ.append(obj)
		full_summary[file] = full_summ

	all_keys = list(set([k for file, full_summ in full_summary.items() for summ in full_summ for k in summ]))
	all_keys = [k for k in all_keys if filt in k and k not in bad_keys]

	xlabel = None

	for k in all_keys:
		plt.figure()
		plt.title(k)
		legend = []
		for file in files:
			subsumm = [summ for summ in full_summary[file] if k in summ]
			if xaxis in ['datetime', 'seconds']:
				subsumm = [summ for summ in subsumm if 'datetime' in summ]
			if k in maxy:
				subsumm = [summ for summ in subsumm if summ[k] < maxy[k]]
			if len(subsumm) == 0: continue

			xs = list(range(len(subsumm)))
			ys = geom([summ[k] for summ in subsumm], geo)
			if xaxis in ['datetime', 'seconds']:
				xs = [summ['datetime'] for summ in subsumm]
				if xaxis == 'seconds':
					start = min(xs)
					xs = np.array([(dt - start).total_seconds() for dt in xs])
					second_span =xs[-1]
					if xlabel is None:
						if second_span > 3*86400:
							xlabel = 'days'
						elif second_span > 5*3600:
							xlabel = 'hours'
						elif second_span > 10*60:
							xlabel = 'minutes'

					if xlabel == 'days': xs /= 86400.0
					if xlabel == 'hours': xs /= 3600.0
					if xlabel == 'minutes': xs /= 60.0

			if xaxis != 'seconds':
				xlabel = xaxis
			legend.append(file.split("/")[-1])
			plt.plot(xs, ys)
			plt.ylabel(k)
			plt.xlabel(xlabel)
		plt.legend(legend)

class LogPlot():

	def __init__(self, where_to):
		# `where_to` the file where you want to save the summaries
		self.current_cache = {}
		self.where_to = where_to

	def cache(self, results, prefix=''):
		# Results should be a dict of keys (of things to save) and values to save {"Loss": 1.0}
		# Prefix: will be added to each key string (for instance a "T" for training, an "V" for validation)

		for k, val in results.items():
			nk = prefix+k
			if nk not in self.current_cache:
				self.current_cache[nk] = []
			self.current_cache[nk].append(float(val))
		return self

	def save(self, printing=False):
		def reduce_array(vals, k):
			if k == "T_count": return sum(vals)
			else:            return float(np.mean(vals))

		save_obj = {k: reduce_array(vals, k) for k, vals in self.current_cache.items()}
		save_obj['datetime'] = str(datetime.now())
		f = open(self.where_to, "a"); f.write(json.dumps(save_obj)+"\n"); f.close()
		self.clear_cache()
		if printing: print(save_obj)

	def clear_cache(self):
		self.current_cache = {} # Reempty the cache
