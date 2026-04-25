# 导入 `regex` 第三方正则库，并起别名为 `re`。
# 这里不用标准库 `re`，是因为 `regex` 对 Unicode 字符类别支持更完整。
import regex as re

# 从 typing 导入可迭代对象和迭代器类型标注。
from typing import Iterable, Iterator


# 定义 Tokenizer 类。
# 它负责：
# 1. 文本 -> token id（encode）
# 2. token id -> 文本（decode）
# 3. 应用 BPE merge 规则
class Tokenizer:
    # 初始化一个 tokenizer。
    def __init__(self, vocab, merges, special_tokens=None):
        # 下面的文档字符串说明三个核心输入。
        '''
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        '''
        # 保存 id -> bytes 的词表映射。
        self.vocab = vocab

        # 保存 BPE merge 规则列表。
        self.merges = merges

        # 保存特殊 token 列表，例如 "<|endoftext|>"。
        self.special_tokens = special_tokens

        # vocab 当前是 id -> bytes。
        # 编码时我们还需要反向查表：bytes -> id。
        self.byte_2_id_vocab = {v: k for k, v in vocab.items()}

        # 把 merge 规则转成 “pair -> 排名” 的字典。
        # 排名越小，说明这条 merge 规则越早、优先级越高。
        self.merges_ranking = {merge: idx for idx, merge in enumerate(merges)}

        # 如果传入了特殊 token，就先把它们编码成 UTF-8 bytes。
        if special_tokens:
            self.special_tokens_2_bytes = [token.encode('UTF-8') for token in special_tokens]

        # 否则就记录成空列表。
        else:
            self.special_tokens_2_bytes = []

        # 把 special tokens 补充进 vocab。
        # 这样后续 encode 时，special token 也能像普通 token 一样映射到唯一 id。
        for special_token in self.special_tokens_2_bytes:
            # 只有当这个特殊 token 还不在词表里时，才追加。
            if special_token not in self.byte_2_id_vocab:
                # 新 token 的 id 就取当前词表长度。
                n = len(self.vocab)

                # 在正向词表中注册新 id -> bytes。
                self.vocab[n] = special_token

                # 在反向词表中注册新 bytes -> id。
                self.byte_2_id_vocab[special_token] = n

    # 定义类方法：从序列化文件中恢复一个 tokenizer。
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        """
        从保存好的 vocab 和 merges 文件中加载 tokenizer。

        Args:
            vocab_filepath: 保存 vocab 的 pickle 文件路径
            merges_filepath: 保存 merge 规则的 pickle 文件路径
            special_tokens: 可选的特殊 token 列表

        Returns:
            一个初始化完成的 Tokenizer 实例
        """
        # 在函数内部导入 pickle，避免模块加载时就增加依赖。
        import pickle

        # 以二进制只读方式打开 vocab 文件。
        with open(vocab_filepath, "rb") as vf:
            # 从 pickle 文件中加载原始 vocab 对象。
            raw_vocab = pickle.load(vf)

        # 准备一个“规范化后的 vocab”字典：
        # key 统一成 int，value 统一成 bytes。
        norm_vocab: dict[int, bytes] = {}

        # 遍历原始 vocab 的每个键值对。
        for k, v in raw_vocab.items():
            # 确保词表 id 一律转成 int。
            kid = int(k)

            # 如果 value 是 str，就转成 UTF-8 bytes。
            if isinstance(v, str):
                v = v.encode("utf-8")

            # 把规范化后的键值对写入新词表。
            norm_vocab[kid] = v

        # 以二进制只读方式打开 merges 文件。
        with open(merges_filepath, "rb") as mf:
            # 从 pickle 文件中加载原始 merge 规则。
            raw_merges = pickle.load(mf)

        # 准备一个“规范化后的 merges”列表。
        norm_merges: list[tuple[bytes, bytes]] = []

        # 遍历原始 merge 规则中的每个 pair。
        for a, b in raw_merges:
            # 如果左半部分是 str，就转成 bytes。
            if isinstance(a, str):
                a = a.encode("utf-8")

            # 如果右半部分是 str，就转成 bytes。
            if isinstance(b, str):
                b = b.encode("utf-8")

            # 把规范化后的 pair 追加到列表里。
            norm_merges.append((a, b))

        # 用规范化后的 vocab、merges 和 special_tokens 构造 tokenizer。
        return cls(norm_vocab, norm_merges, special_tokens)

    # 定义编码函数：把整段字符串编码成 token id 列表。
    def encode(self, text: str) -> list[int]:
        # 准备最终返回的 token id 列表。
        res_token_ids = []

        # 先做 pre-tokenization，把原始文本切成若干较小片段。
        pretokenizeiton = self.pre_tokenization(text, self.special_tokens)

        # 逐段处理每个 pre-token。
        for part in pretokenizeiton:
            # 如果当前片段本身就是 special token，就直接查它对应的 id。
            if self.special_tokens and part in self.special_tokens:
                # 先把 special token 字符串转成 bytes，再去反向词表查 id。
                special_id = self.byte_2_id_vocab[part.encode('UTF-8')]

                # 把这个 special token 的 id 追加到结果里。
                res_token_ids.append(special_id)

            # 否则，这只是普通文本片段，需要继续做 BPE 编码。
            else:
                # encode_text 返回的是一段片段对应的多个 token id，
                # 所以这里用 extend 展开追加。
                res_token_ids.extend(self.encode_text(part))

        # 返回整段文本的最终编码结果。
        return res_token_ids

    # 定义流式编码函数：输入一个字符串可迭代对象，逐块产出 token id。
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # 依次读取每个文本块。
        for chunk in iterable:
            # 对每个文本块调用 encode，再把生成的 token id 一个个 yield 出去。
            yield from self.encode(chunk)

    # 定义解码函数：把 token id 列表还原成字符串。
    def decode(self, ids: list[int]) -> str:
        # 先按顺序把每个 id 对应的 bytes 片段拼接起来。
        byte_list = b''.join(self.vocab[id_] for id_ in ids)

        # 再把整段 bytes 用 UTF-8 解码成字符串。
        # errors='replace' 表示遇到非法字节时用替代字符兜底。
        return byte_list.decode('UTF-8', errors='replace')

    # 定义“普通文本片段”的编码函数。
    # 这里的输入已经不是整段文本，而是一个 pre-token。
    def encode_text(self, pre_token: str):
        '''
        encode a single pre-token (normal text, not special tokens) to token ids
        '''
        # 定义内部辅助函数：把一个字符串拆成逐字节的 bytes token。
        def word_2_byte(word: str) -> tuple[bytes, ...]:
            # 先把字符串编码成 UTF-8 字节序列，再转成整数列表。
            word_decoded = list(word.encode('UTF-8'))

            # 把每个单独的整数 byte 再包装成长度为 1 的 bytes 对象。
            word_byte = [bytes([b]) for b in word_decoded]

            # 返回不可变的 tuple，表示最初始的逐字节 token 序列。
            return tuple(word_byte)

        # 把当前 pre-token 转成逐字节 token 序列。
        word_byte = word_2_byte(pre_token)

        # 应用 BPE merge 规则，把小 bytes token 合并成更大的子词 token。
        word_byte_after_merge = self.apply_merge(word_byte)

        # 准备存放合并后 token 对应的 id。
        token_ids = []

        # 遍历每个合并后的 bytes token。
        for merged_bytes in word_byte_after_merge:
            # 在反向词表里查到这个 bytes token 对应的 id。
            id_ = self.byte_2_id_vocab[merged_bytes]

            # 把 id 追加到结果列表中。
            token_ids.append(id_)

        # 返回这个 pre-token 对应的一串 token id。
        return token_ids

    # 定义 BPE merge 核心函数。
    def apply_merge(self, word_byte):
        # 先把输入转成 list，方便后续原地式重建 token 序列。
        word = list(word_byte)

        # 定义内部函数：提取相邻 token 两两组成的 pair 集合。
        def get_pairs(word):
            # 用集合存 pair，避免重复。
            pairs = set()

            # 先把第一个 token 记作前一个字符。
            prev_char = word[0]

            # 从第二个 token 开始遍历。
            for char in word[1:]:
                # 记录相邻 pair：(前一个 token, 当前 token)。
                pairs.add((prev_char, char))

                # 更新“前一个 token”。
                prev_char = char

            # 返回当前 token 序列里所有相邻 pair。
            return pairs

        # 先抽取当前 token 序列中的所有相邻 pair。
        word_pairs = get_pairs(word)

        # 如果根本没有 pair，说明长度不足 2，直接返回。
        if not word_pairs:
            return word

        # 反复应用 merge，直到没有可合并 pair。
        while True:
            # 从当前所有 pair 中，找出“merge 排名最靠前”的那个 bigram。
            # 如果某个 pair 不在 merges_ranking 中，就给它无穷大排名，表示不可 merge。
            bigram = min(word_pairs, key=lambda pair: self.merges_ranking.get(pair, float('inf')))

            # 如果当前最优 bigram 也不在 merge 规则中，说明不能再合并了。
            if bigram not in self.merges_ranking:
                break

            # 用 idx 表示当前扫描到 word 的哪个位置。
            idx = 0

            # 用新列表承接本轮 merge 后的 token 序列。
            new_byte_token = []

            # 把当前要合并的 pair 解包成 first 和 second。
            first, second = bigram

            # 从左到右扫描整个 token 序列。
            while idx < len(word):
                try:
                    # 从 idx 开始，找最近的一个 first。
                    first_nearest = word.index(first, idx)

                # 如果后面再也找不到 first，就把剩余部分原样追加并结束这一轮扫描。
                except ValueError:
                    new_byte_token.extend(word[idx:])
                    break

                # 找到了 first，就进入这里。
                else:
                    # 把 idx 到 first_nearest 之间那些不相关 token 原样复制过去。
                    new_byte_token.extend(word[idx:first_nearest])

                    # 把扫描指针移动到找到的 first 上。
                    idx = first_nearest

                # 如果当前位置确实是 first，并且下一个位置存在且正好是 second，
                # 就把这两个 token 合并成一个更大的 bytes token。
                if word[first_nearest] == first and first_nearest + 1 < len(word) and word[first_nearest + 1] == second:
                    # 把 first+second 这个合并后的 token 放入新序列。
                    new_byte_token.append(first + second)

                    # 因为已经吃掉了两个 token，所以指针前进 2。
                    idx += 2

                # 否则说明这里只匹配到了 first，但没形成完整 bigram。
                else:
                    # 把这个 first 原样放回去。
                    new_byte_token.append(word[first_nearest])

                    # 只消耗一个 token。
                    idx += 1

            # 用本轮 merge 后的新 token 序列替换旧序列。
            word = new_byte_token

            # 如果已经只剩一个 token，就不可能再有 pair，直接结束。
            if len(word) == 1:
                break

            # 否则重新计算新序列中的相邻 pair，准备下一轮 merge。
            else:
                word_pairs = get_pairs(word)

        # 返回最终 merge 完成后的 bytes token 序列。
        return word

    # 定义预切分函数：先把原始字符串切成更小的“预 token”片段。
    def pre_tokenization(self, s: str, special_token: list[str]) -> list[str]:
        # 定义 GPT 风格的预分词正则：
        # - 英文缩写
        # - 字母串
        # - 数字串
        # - 标点/符号串
        # - 空白串
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # 如果没有 special token，就直接按 PAT 做普通正则切分。
        if not special_token:
            return re.findall(PAT, s)

        # 如果有 special token，就先按长度从长到短排序。
        # 这样可以避免短 special token 抢先匹配长 special token 的前缀。
        toks = sorted(special_token, key=len, reverse=True)

        # 把所有 special token 转义后用 | 拼成一个联合正则。
        union = "|".join(re.escape(t) for t in toks)

        # 用 re.split 把原始字符串按 special token 边界切开。
        # 因为用了捕获组 (...)，所以 special token 本身也会被保留下来。
        parts = re.split(f"({union})", s)

        # 准备最终输出列表。
        out = []

        # 把 special token 列表转成集合，方便 O(1) 判断。
        st = set(special_token)

        # 逐段处理 split 后的每个片段。
        for part in parts:
            # 空串没有意义，直接跳过。
            if not part:
                continue

            # 如果当前片段本身就是一个完整 special token，
            # 就原样保留，不再继续拆分。
            if part in st:
                out.append(part)

            # 否则，这只是普通文本，需要继续用 PAT 做细粒度切分。
            else:
                out.extend(re.findall(PAT, part))

        # 返回最终 pre-tokenization 结果。
        return out
