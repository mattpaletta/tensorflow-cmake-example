#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"

using namespace tensorflow;
using namespace tensorflow::ops;

int main() {
    /* // Enable XLA
    * OptimizerOptions optimizer_options = _tf_options.config.graph_options().optimizer_options();
    * optimizer_options.set_global_jit_level(OptimizerOptions::ON_1);
    * GraphOptions graph_options = _tf_options.config.graph_options();
    * graph_options.mutable_optimizer_options()->CopyFrom(optimizer_options);
    * _tf_options.config.mutable_graph_options()->CopyFrom(graph_options);
    * */

    Scope root = Scope::NewRootScope();
    // Matrix A = [3 2; 
    //            -1 0]
    auto A = Const(root, { {3.f, 2.f}, {-1.f, 0.f}});
    
    // Vector b = [3 5]
    auto b = Const(root, { {3.f, 5.f}});
    
    // v = Ab^T
    auto v = MatMul(root.WithOpName("v"), A, b, MatMul::TransposeB(true));
    std::vector<Tensor> outputs;
    
    ClientSession session(root);
    
    // Run and fetch v
    TF_CHECK_OK(session.Run({v}, &outputs));
    
    // Expect outputs[0] == [19; -3]
    LOG(INFO) << outputs[0].matrix<float>();
    std::cout << "compiled correctly!" << std::endl;
    return 0;
}
