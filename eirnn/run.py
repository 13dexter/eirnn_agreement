from agreement_acceptor import PredictVerbNumber
import filenames
pvn = PredictVerbNumber(filenames.deps, prop_train=0.1)
pvn.pipeline()