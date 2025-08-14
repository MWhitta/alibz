# Save first line for Fortuna
import socketio
import numpy as np
import eventlet
from flask import Flask, request, jsonify
import argparse
from peaky_finder import PeakyFinder
from peaky_indexer import PeakyIndexer
import pandas as pd

class AlibzSocketIOServer():
    """
    A socket.io server class for alibz analysis.
    """
    def __init__(self):
        
        self.sio = socketio.Server(cors_allowed_origins='*', async_mode='eventlet')
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
        # self.sio.on('measure', self.on_measure)
        # self.sio.on('export', self.on_export)
        # self.sio.on('analyze', self.on_analyze)
        # self.sio.on('find_buttons', self.on_find_buttons)
        # self.sio.on('change_export_path', self.on_change_export_path)

        self.peaky_finder = PeakyFinder('spectra')
        self.peaky_indexer = PeakyIndexer(self.peaky_finder)

    def on_connect(self, sid, environ, auth):
        print('connect ', sid)

    def on_disconnect(self, sid):
        print('disconnect ', sid)

    def on_one_click(self, sid, params: dict):
        print(f'received one click request from {sid}.')
        x, y, n_sigma, subtract_background = np.array(params['wavelength']), np.array(params['intensity']), params['n_sigma'], params['subtract_background']
    
        # Spawn a greenlet to handle the potentially blocking operation
        eventlet.spawn(self._process_one_click, sid, x, y, n_sigma, subtract_background)
    
    def _process_one_click(self, sid, x, y, n_sigma, subtract_background):
        """Process the one_click request in a separate greenlet to avoid blocking"""
        try:
            finder_dict = self.peaky_finder.fit_spectrum(x, y, n_sigma, subtract_background, plot=True)
            # Emit the result back to the client
            self.sio.emit('one_click_result', finder_dict, room=sid)
        except Exception as e:
            print(f"Error processing one_click for {sid}: {e}")
            # Emit error back to the client
            self.sio.emit('one_click_error', {'error': str(e)}, room=sid)

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
    eventlet.wsgi.server(eventlet.listen(('', 2518)), alibz_web_server.app)


