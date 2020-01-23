import matlab.engine

eng = matlab.engine.start_matlab()
eng.spectra_generator_simple(nargout=0)

eng.quit()

