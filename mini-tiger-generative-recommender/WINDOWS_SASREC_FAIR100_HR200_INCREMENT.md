# SASRec V2 公平目标增量重训

将本增量包覆盖到之前的 SASRec V2 Windows 完整包。本次使用新配置和新输出目录，
不会续接或覆盖旧实验。

## 对齐内容

- 与当前 MiniTIGER 一样，每用户只监督最后 100 个正目标；
- 预计训练目标约 71.6 万个；
- 每个目标仍使用此前最多 100 条完整正负交互作为上下文；
- Feedback Type Embedding、正目标切分和不屏蔽历史物品保持不变；
- 统一用 Validation HitRate@200 早停，patience=10；
- 最终输出 @5、10、20、50、100、200、500 和 AUC/UAUC。

## 运行

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_sasrec_fair100_hr200.ps1
```

结果位于：

```text
outputs/kuairec_big_sasrec_v2_fair100_hr200_cuda/
```

中断后重复同一命令可从这个新实验的检查点继续训练。
