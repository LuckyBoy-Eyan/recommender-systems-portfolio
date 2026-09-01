# Windows 双塔 V2 四折 OOF

将本增量包覆盖到原项目根目录，然后在 PowerShell 执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_windows_two_tower_v2_oof.ps1
```

脚本按严格时间顺序训练四个独立模型。每折只使用 cutoff 之前的数据训练，并只给随后时间段生成 Top300；不读取测试集。为避免用 OOF 标签选择模型，四折均固定训练 6 轮，不做外层验证早停。

完成后传回：

`outputs/windows_two_tower_v2_oof_results.zip`

若中途停止，已完成折的目录可以保留；再次运行时请把已完成折单独备份，当前脚本会重新执行对应折。
