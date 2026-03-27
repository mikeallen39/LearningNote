import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 核心层实现 (修复版)
# ==========================================

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))
        self.cache = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        self.cache = X
        return X @ self.W + self.b

    def backward(self, grad_output):
        X = self.cache
        # ⭐ 修复：不再除以 N，因为 loss_fn.backward 已经除过了
        self.grad_W = X.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.W.T

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = (X > 0)
        return X * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask

class SoftmaxCrossEntropy:
    def __init__(self):
        self.probs = None
        self.labels = None

    def forward(self, X, y):
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.labels = y
        N = X.shape[0]
        log_probs = -np.log(self.probs[np.arange(N), y] + 1e-8)
        # Loss 平均
        loss = np.sum(log_probs) / N
        return loss

    def backward(self):
        N = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.labels] -= 1
        # ⭐ 梯度平均 (这样 Linear.backward 就不需要再除了)
        return grad / N

# ==========================================
# 2. 梯度检查工具 (修复版)
# ==========================================

def gradient_check(layer, X, y, loss_fn, param_name='W', eps=1e-5, tol=1e-4):
    print(f"--- 正在检查 {layer.__class__.__name__}.{param_name} 的梯度 ---")
    
    # 1. 计算解析梯度
    out = layer.forward(X)
    loss = loss_fn.forward(out, y)
    grad_output = loss_fn.backward()
    layer.backward(grad_output)
    
    if param_name == 'W':
        grad_analytical = layer.grad_W.flatten()
        param = layer.W
    elif param_name == 'b':
        grad_analytical = layer.grad_b.flatten()
        param = layer.b
    
    # 2. 计算数值梯度
    grad_numerical = np.zeros_like(param).flatten()
    original_param = param.copy()
    
    for i in range(len(grad_numerical)):
        idx = np.unravel_index(i, param.shape)
        
        # W + eps
        param[idx] = original_param[idx] + eps
        out_plus = layer.forward(X)
        loss_plus = loss_fn.forward(out_plus, y)
        
        # W - eps
        param[idx] = original_param[idx] - eps
        out_minus = layer.forward(X)
        loss_minus = loss_fn.forward(out_minus, y)
        
        grad_numerical[i] = (loss_plus - loss_minus) / (2 * eps)
    
    # 恢复参数
    param[:] = original_param
    
    # 3. 计算相对误差
    diff = np.linalg.norm(grad_analytical - grad_numerical) / \
           (np.linalg.norm(grad_analytical) + np.linalg.norm(grad_numerical) + 1e-8)
    
    if diff < tol:
        print(f"✅ 梯度检查通过！差异值：{diff:.2e}")
        return True
    else:
        print(f"❌ 梯度检查失败！差异值：{diff:.2e} (阈值：{tol})")
        return False

# ==========================================
# 3. 数据生成
# ==========================================

def generate_xor_data(N=100):
    X = np.random.randn(N, 2)
    y = np.zeros(N, dtype=int)
    mask = (X[:, 0] * X[:, 1] < 0)
    y[mask] = 1
    return X, y

# ==========================================
# 4. 训练演示
# ==========================================

def train_demo():
    print("\n=== 开始训练演示 (MLP on XOR) ===")
    np.random.seed(42)
    
    X, y = generate_xor_data(N=200)
    
    fc1 = Linear(2, 10)
    relu1 = ReLU()
    fc2 = Linear(10, 2)
    loss_fn = SoftmaxCrossEntropy()
    
    lr = 0.5
    epochs = 500
    loss_history = []
    
    # 梯度检查
    h = relu1.forward(fc1.forward(X))
    gradient_check(fc2, h, y, loss_fn, param_name='W')
    gradient_check(fc2, h, y, loss_fn, param_name='b')
    
    # 训练循环
    for epoch in range(epochs):
        h = fc1.forward(X)
        h = relu1.forward(h)
        logits = fc2.forward(h)
        loss = loss_fn.forward(logits, y)
        
        grad = loss_fn.backward()
        grad = fc2.backward(grad)
        grad = relu1.backward(grad)
        grad = fc1.backward(grad)
        
        for layer in [fc1, fc2]:
            layer.W -= lr * layer.grad_W
            layer.b -= lr * layer.grad_b
            
        loss_history.append(loss)
        
        if epoch % 50 == 0:
            preds = np.argmax(logits, axis=1)
            acc = np.mean(preds == y)
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Acc: {acc:.2f}")
            
    # 绘图
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=100)
    print("\nLoss 曲线已保存为 loss_curve.png")
    print("=== 训练结束 ===")

if __name__ == "__main__":
    np.random.seed(123)
    test_layer = Linear(5, 3)
    test_X = np.random.randn(10, 5)
    test_y = np.random.randint(0, 3, 10)
    test_loss = SoftmaxCrossEntropy()
    
    print("=" * 50)
    gradient_check(test_layer, test_X, test_y, test_loss, param_name='W')
    gradient_check(test_layer, test_X, test_y, test_loss, param_name='b')
    print("=" * 50)
    
    train_demo()