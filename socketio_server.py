# Save first line for Fortuna
import socketio
import numpy as np
from flask import Flask, request, jsonify
import argparse
from peaky_finder import PeakyFinder
from peaky_indexer import PeakyIndexer
from peaky_maker import PeakyMaker
import pandas as pd
import matplotlib.pyplot as plt
from gevent import pywsgi
import gevent
import threading

# Create a dictionary converting element abbreviations to their index
ELEMENTS = [
                'H', 'He',  # Row 1
                'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',  # Row 2
                'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',  # Row 3
                'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',  # Row 4  # Se removed between As/Br
                'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',  # Row 5
                'Cs', 'Ba',  # Row 6 alkali/alkaline earth
                'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',  # Row 6 rare earths  # Pm removed between Nd/Sm
                'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',  # Row 6 transition metals  # Po,At,Rn removed between Bi/Rn
                'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U',  # Th,Pa,U removed
            ]
ELEMENT_ABBR_TO_INDEX = {el: idx for idx, el in enumerate(ELEMENTS)}
ELEMENT_INDEX_TO_ABBR = {idx: el for idx, el in enumerate(ELEMENTS)}


class AlibzSocketIOServer():
    """
    A socket.io server class for alibz analysis.
    """
    def __init__(self):
        
        self.sio = socketio.Server(cors_allowed_origins='*', async_mode='gevent')
        # Initialize Flask app for HTTP endpoints
        self.flask_app = Flask(__name__)
        self.app = socketio.WSGIApp(self.sio, self.flask_app)

        # Register HTTP routes
        self.flask_app.route('/one_click', methods=['GET'])(self.http_one_click)
        # self.flask_app.route('/export', methods=['POST'])(self.http_export)
        # self.flask_app.route('/analyze', methods=['POST'])(self.http_analyze)
        # self.flask_app.route('/find_buttons', methods=['GET'])(self.http_find_buttons)
        # self.flask_app.route('/change_export_path', methods=['POST'])(self.http_change_export_path)
        # self.flask_app.route('/status', methods=['GET'])(self.http_status)
    
        self.sio.on('connect', self.on_connect)
        self.sio.on('disconnect', self.on_disconnect)
        self.sio.on('one_click', self.on_one_click)
        self.sio.on('sim_peak', self.on_sim_peak)
        # self.sio.on('measure', self.on_measure)
        # self.sio.on('export', self.on_export)
        # self.sio.on('analyze', self.on_analyze)
        # self.sio.on('find_buttons', self.on_find_buttons)
        # self.sio.on('change_export_path', self.on_change_export_path)

        self.peaky_finder = PeakyFinder('spectra')
        self.peaky_indexer = PeakyIndexer(self.peaky_finder)
        self.peaky_maker = PeakyMaker('db')

    def on_connect(self, sid, environ, auth):
        print('connect ', sid)

    def on_disconnect(self, sid):
        print('disconnect ', sid)

    def on_one_click(self, sid, params: dict):
        print(f'received one click request from {sid}.')
        x, y, n_sigma, subtract_background, elements = np.array(params['wavelength']), np.array(params['intensity']), params['n_sigma'], params['subtract_background'], params['elements']
        
        # Use standard Python threading to handle the processing
        thread = threading.Thread(target=self._process_one_click, args=(sid, x, y, n_sigma, subtract_background, elements))
        thread.daemon = True  # Make thread daemon so it doesn't block server shutdown
        thread.start()
    def _process_one_click(self, sid, x, y, n_sigma, subtract_background, elements):
        """Process the one_click request in a separate greenlet to avoid blocking"""
        try:
            finder_dict = self.peaky_finder.fit_spectrum(x, y, n_sigma, subtract_background, plot=False)
            test_dict = self.peaky_indexer.peak_match(finder_dict['sorted_parameter_array'], element_list=elements)
            test_locs = np.array(list(test_dict['Li'].ions[1.0].values()))
            plt.scatter(test_locs[:,0], test_locs[:,1])
            plt.show()
            # Emit the result back to the client
            self.sio.emit('one_click_result', finder_dict, room=sid)
        except Exception as e:
            print(f"Error processing one_click for {sid}: {e}")
            # Emit error back to the client
            self.sio.emit('one_click_error', {'error': str(e)}, room=sid)

    def on_sim_peak(self, sid, params: dict):
        try:
            print(f'received sim_peak request from {sid}.')
            # elements = params['elements']
            # fracs = [0] * self.peaky_maker.max_z
            # for el in elements:
            #     fracs[ELEMENT_ABBR_TO_INDEX[el]] = 1
            # fracs = np.array(fracs)
            # wavelength, spectrum, element_spectra = self.peaky_maker.peak_maker(fracs)
            # ele_spectra = {el: element_spectra[ELEMENT_ABBR_TO_INDEX[el]].tolist() for el in elements}
            # print(ele_spectra)
            # print(spectrum)
            # print(ele_spectra)
            # self.sio.emit('sim_peak_result', {'wavelength': wavelength.tolist(), 'spectrum': spectrum.tolist(), 'element_spectra': ele_spectra}, room=sid)
            import pandas as pd
            df = pd.read_csv('./spectra/2025_06_23_14_55_42/2025_06_23_14_55_42_20250623_025539_PM_Spectrum1.csv', header=0, index_col=None)
            wavelength = df['wavelength'].values
            spectrum = df['intensity'].values
            self.sio.emit('sim_peak_result', {'wavelength': wavelength.tolist(), 'spectrum': spectrum.tolist(), 'element_spectra': {}}, room=sid)
        except Exception as e:
            print(f"Error processing sim_peak for {sid}: {e}")
            # Emit error back to the client
            self.sio.emit('sim_peak_error', {'error': str(e)}, room=sid)
            raise(e)

    def http_one_click(self):
        pass
    

    # # HTTP endpoint handlers
    # def http_measure(self):
    #     """HTTP endpoint for measure operation"""
    #     if self.status == AnalyzerStatus.RUNNING:
    #         return jsonify({'error': 'The analyzer is currently running. Please wait until it is done.'}), 409
    #     else:
    #         try:
    #             self.measure()
    #             return jsonify({'message': 'Measurement completed successfully'}), 200
    #         except Exception as e:
    #             print(e)
    #             return jsonify({'error': str(e)}), 500
    

    # def http_status(self):
    #     """HTTP endpoint for getting current status"""
    #     return jsonify({'status': self.status.name}), 200

    # def update_status(self):
    #     while True:
    #         self.sio.sleep(0.5)
    #         self.sio.emit('status', self.status.name)
    #         self.sio.emit('export_path', self.export_folder_path)
        

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Z300SocketIOServer")
    # parser.add_argument('export_folder_path', type=str, help='Path to the export folder, e.g. C:/Users/Whittaker/Documents/20250623Scanning')
    # args = parser.parse_args()
    
    alibz_web_server = AlibzSocketIOServer()
    # z300_web_server.sio.start_background_task(z300_web_server.update_status)
    # eventlet.wsgi.server(eventlet.listen(('', 2518)), alibz_web_server.app)
    pywsgi.WSGIServer(('', 2518), alibz_web_server.app).serve_forever()


