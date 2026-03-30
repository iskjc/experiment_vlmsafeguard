# 激活重计算，用时间换显存
if hasattr(self._lang_backbone, 'gradient_checkpointing_enable'):
    self._lang_backbone.gradient_checkpointing_enable()
