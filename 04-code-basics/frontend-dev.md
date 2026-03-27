# 前端开发笔记

---

## JWT Token 认证

### 传统 Token 认证流程
1. 客户端发送用户名密码，服务器创建 Token 并存储
2. 返回 Token 给客户端
3. 客户端每次请求携带 Token
4. 服务器查库验证 Token

**缺点**：每次验证需要查库；退出登录需要通知服务器删除 Token。

### JWT 认证流程
1. 服务器创建 Token 返回给客户端，自己不存储
2. 客户端保存 Token，每次请求携带
3. 服务器直接解析 Token，不查库

**优势**：服务器不存储 Token，不用查库；分布式系统中每个服务器都可以独立验证。

### HTTP 认证方式对比

| 方式 | 特点 | 安全性 |
|------|------|--------|
| HTTP Basic | Base64 编码用户名密码 | 不安全，明文传输 |
| Session-Cookie | 服务端存储 Session | 需配合 HTTPS |
| Token/JWT | 无状态，客户端存储 | 较安全 |

### Cocos Creator 环境注意事项

Cocos Creator 不支持 Node.js 的 `jsonwebtoken` 库，可使用 `jwt-decode` 解码（但不支持签名验证）。

---

## 微信小游戏开发

### 分享功能

```typescript
window['wx'].shareAppMessage({
  title: '分享标题',
  imageUrl: 'path/to/image.jpg',
  query: 'key1=val1&key2=val2',
  success: function () { },
  fail: function () { },
  complete: function () { }
});
```

### 防沉迷

参考：[微信小游戏防沉迷文档](https://developers.weixin.qq.com/community/minigame/doc/000424cae449405268f9debd156c00)

### 动态更换 Sprite

```typescript
changeSprite(node: Node, iconPath: string, height: number, width: number) {
    resources.load(iconPath + '/spriteFrame', SpriteFrame, (err, spriteFrame: SpriteFrame) => {
        if (!err) {
            node.getComponent(UITransform).setContentSize(height, width);
            node.getComponent(Sprite).spriteFrame = spriteFrame;
        }
    })
}
```

---
