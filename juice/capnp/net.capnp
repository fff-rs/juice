# Types required for serialization of a Network.

@0xd42882f7f90ce348;

struct SequentialConfig {
}

struct LinearConfig {
    outputSize @0 :UInt64;
}

struct NegativeLogLikelihoodConfig {
}

struct LayerConfig {
    layerType :union {
        sequential @0 :SequentialConfig; 
        linear @1 :LinearConfig;
        logSoftmax @2 :Void;
        relu @3 :Void;
        sigmoid @4 :Void;
        meanSquaredError @5 :Void;
        negativeLogLikelihood @6 :NegativeLogLikelihoodConfig;
    }
}

struct TensorShape {
    shape @0 :List(UInt64);
}

struct Network {
    config @0 :LayerConfig;
    inputs @1 :List(TensorShape);
}