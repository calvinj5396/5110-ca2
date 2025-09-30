import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings('ignore')


class WideDeepModel:
    """Wide & Deep模型实现"""

    def __init__(self, wide_features, deep_features, deep_hidden_units=[128, 64, 32],
                 dropout_rate=0.2, learning_rate=0.001):
        """
        初始化Wide & Deep模型

        Args:
            wide_features: Wide部分的特征配置列表
            deep_features: Deep部分的特征配置列表
            deep_hidden_units: Deep部分隐层单元数
            dropout_rate: Dropout比例
            learning_rate: 学习率
        """
        self.wide_features = wide_features
        self.deep_features = deep_features
        self.deep_hidden_units = deep_hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None

    def build_model(self):
        """构建Wide & Deep模型"""

        # 输入层
        inputs = {}

        # Wide部分输入
        wide_inputs = []
        for feature in self.wide_features:
            if feature['type'] == 'categorical':
                input_layer = keras.Input(shape=(1,), name=f"wide_{feature['name']}")
                inputs[f"wide_{feature['name']}"] = input_layer

                # 对categorical特征进行embedding后再线性变换
                embedding = layers.Embedding(
                    input_dim=feature['vocab_size'],
                    output_dim=1,  # Wide部分使用1维embedding
                    name=f"wide_emb_{feature['name']}"
                )(input_layer)
                wide_inputs.append(layers.Flatten()(embedding))

            elif feature['type'] == 'numerical':
                input_layer = keras.Input(shape=(1,), name=f"wide_{feature['name']}")
                inputs[f"wide_{feature['name']}"] = input_layer
                wide_inputs.append(input_layer)

        # Deep部分输入
        deep_inputs = []
        for feature in self.deep_features:
            if feature['type'] == 'categorical':
                input_layer = keras.Input(shape=(1,), name=f"deep_{feature['name']}")
                inputs[f"deep_{feature['name']}"] = input_layer

                # Categorical特征使用embedding
                embedding = layers.Embedding(
                    input_dim=feature['vocab_size'],
                    output_dim=feature['embedding_dim'],
                    name=f"deep_emb_{feature['name']}"
                )(input_layer)
                deep_inputs.append(layers.Flatten()(embedding))

            elif feature['type'] == 'numerical':
                input_layer = keras.Input(shape=(1,), name=f"deep_{feature['name']}")
                inputs[f"deep_{feature['name']}"] = input_layer
                deep_inputs.append(input_layer)

        # Wide部分：线性模型
        if wide_inputs:
            wide_concat = layers.Concatenate()(wide_inputs) if len(wide_inputs) > 1 else wide_inputs[0]
            wide_output = layers.Dense(1, activation=None, name='wide_output')(wide_concat)
        else:
            wide_output = layers.Dense(1, activation=None)(layers.Input(shape=(1,)))

        # Deep部分：多层感知器
        if deep_inputs:
            deep_concat = layers.Concatenate()(deep_inputs) if len(deep_inputs) > 1 else deep_inputs[0]

            deep_layer = deep_concat
            for i, units in enumerate(self.deep_hidden_units):
                deep_layer = layers.Dense(units, activation='relu', name=f'deep_layer_{i}')(deep_layer)
                deep_layer = layers.Dropout(self.dropout_rate)(deep_layer)

            deep_output = layers.Dense(1, activation=None, name='deep_output')(deep_layer)
        else:
            deep_output = layers.Dense(1, activation=None)(layers.Input(shape=(1,)))

        # 组合Wide和Deep的输出
        combined = layers.Add()([wide_output, deep_output])
        output = layers.Dense(1, activation='sigmoid', name='prediction')(combined)

        # 创建模型
        self.model = keras.Model(inputs=list(inputs.values()), outputs=output)

        # 编译模型
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc']
        )

        return self.model

    def train(self, train_data, validation_data=None, epochs=10, batch_size=256, verbose=1):
        """训练模型"""
        if self.model is None:
            self.build_model()

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2)
        ]

        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def predict(self, test_data, batch_size=256):
        """预测"""
        return self.model.predict(test_data, batch_size=batch_size)

    def evaluate(self, test_data, batch_size=256):
        """评估模型"""
        return self.model.evaluate(test_data, batch_size=batch_size)


class FeatureProcessor:
    """特征处理器"""

    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}

    def process_features(self, df, categorical_cols, numerical_cols, fit=True):
        """处理特征"""
        processed_data = {}

        # 处理categorical特征
        for col in categorical_cols:
            if fit:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                processed_data[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                processed_data[col] = self.label_encoders[col].transform(df[col].astype(str))

        # 处理numerical特征
        for col in numerical_cols:
            if fit:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                processed_data[col] = self.scalers[col].fit_transform(df[[col]]).flatten()
            else:
                processed_data[col] = self.scalers[col].transform(df[[col]]).flatten()

        return processed_data

    def get_vocab_sizes(self, categorical_cols):
        """获取categorical特征的词汇表大小"""
        vocab_sizes = {}
        for col in categorical_cols:
            if col in self.label_encoders:
                vocab_sizes[col] = len(self.label_encoders[col].classes_)
        return vocab_sizes


def create_sample_data(n_samples=10000):
    """创建示例数据集"""
    np.random.seed(42)

    # 生成特征
    data = {
        'user_id': np.random.randint(1, 1000, n_samples),
        'item_id': np.random.randint(1, 500, n_samples),
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples),
        'price': np.random.uniform(10, 1000, n_samples),
        'user_age': np.random.randint(18, 65, n_samples),
        'item_popularity': np.random.uniform(0, 1, n_samples)
    }

    df = pd.DataFrame(data)

    # 生成标签（简单的规则，实际应用中会更复杂）
    df['label'] = (
            (df['category'].isin(['A', 'B'])) * 0.3 +
            (df['price'] < 100) * 0.2 +
            (df['user_age'] < 30) * 0.2 +
            (df['item_popularity'] > 0.5) * 0.3 +
            np.random.uniform(0, 0.2, n_samples)
    )
    df['label'] = (df['label'] > 0.5).astype(int)

    return df


def main():
    """主函数演示"""

    # 1. 创建示例数据
    print("创建示例数据...")
    df = create_sample_data(10000)
    print(f"数据形状: {df.shape}")
    print(f"标签分布: {df['label'].value_counts()}")

    # 2. 定义特征
    categorical_cols = ['user_id', 'item_id', 'category']
    numerical_cols = ['price', 'user_age', 'item_popularity']

    # 3. 特征处理
    processor = FeatureProcessor()

    # 分割数据
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    # 处理训练数据
    train_features = processor.process_features(train_df, categorical_cols, numerical_cols, fit=True)
    train_labels = train_df['label'].values

    # 处理测试数据
    test_features = processor.process_features(test_df, categorical_cols, numerical_cols, fit=False)
    test_labels = test_df['label'].values

    # 获取词汇表大小
    vocab_sizes = processor.get_vocab_sizes(categorical_cols)

    # 4. 定义模型特征配置
    wide_features = []
    deep_features = []

    # Wide特征配置
    for col in categorical_cols:
        wide_features.append({
            'name': col,
            'type': 'categorical',
            'vocab_size': vocab_sizes[col]
        })

    for col in numerical_cols:
        wide_features.append({
            'name': col,
            'type': 'numerical'
        })

    # Deep特征配置
    for col in categorical_cols:
        embedding_dim = min(50, vocab_sizes[col] // 2) + 1
        deep_features.append({
            'name': col,
            'type': 'categorical',
            'vocab_size': vocab_sizes[col],
            'embedding_dim': embedding_dim
        })

    for col in numerical_cols:
        deep_features.append({
            'name': col,
            'type': 'numerical'
        })

    # 5. 构建和训练模型
    print("\n构建Wide & Deep模型...")
    model = WideDeepModel(
        wide_features=wide_features,
        deep_features=deep_features,
        deep_hidden_units=[128, 64, 32],
        dropout_rate=0.3,
        learning_rate=0.001
    )

    # 构建模型
    keras_model = model.build_model()
    print(keras_model.summary())

    # 准备输入数据
    train_input = {}
    test_input = {}

    for col in categorical_cols + numerical_cols:
        train_input[f'wide_{col}'] = train_features[col]
        train_input[f'deep_{col}'] = train_features[col]
        test_input[f'wide_{col}'] = test_features[col]
        test_input[f'deep_{col}'] = test_features[col]

    # 训练模型
    print("\n开始训练...")
    history = model.train(
        train_data=(train_input, train_labels),
        validation_data=(test_input, test_labels),
        epochs=20,
        batch_size=256,
        verbose=1
    )

    # 6. 评估模型
    print("\n评估模型...")
    test_loss, test_accuracy, test_auc = model.evaluate((test_input, test_labels))
    print(f"测试集 - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f}")

    # 7. 预测
    predictions = model.predict((test_input, test_labels))
    print(f"\n预测示例 (前10个): {predictions[:10].flatten()}")

    return model, history


if __name__ == "__main__":
    model, history = main()