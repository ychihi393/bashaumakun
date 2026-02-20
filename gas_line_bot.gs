// ================================================================
// LINE Bot + LSTEP プロキシ — Google Apps Script
// ================================================================
// 設定値（変更不要）
var LINE_CHANNEL_ACCESS_TOKEN = '6JrvKVWm9jju3h6k6ZLlb016vQrv7igk5ZBdkWS4caFRQRuVJOo3fVCkOJ+ODhgHZ0BcAPeuW0MfoOkNSvwjz6eP4c8v6eqUsdGkp+zVBhLeeH4L1puagLw2u+7gP1EnJY6IIwFV9a5LYrQBj6P16QdB04t89/1O/w1cDnyilFU=';
var LSTEP_WEBHOOK_URL         = 'https://rcv.linestep.net/v3/call/2008591924';
var UPDATE_SECRET             = '85c50a5ec03255c188dad68d440f92ec'; // main.py と共有

// ================================================================
// エントリーポイント
// ================================================================
function doPost(e) {
  var bodyText = e.postData.contents;
  var body;
  try { body = JSON.parse(bodyText); } catch (err) {
    return ContentService.createTextOutput('Bad Request');
  }

  // ── main.py からの物件データ更新 ────────────────────────────────
  if (body.type === 'update_properties') {
    if (body.secret !== UPDATE_SECRET) {
      return ContentService.createTextOutput('Unauthorized');
    }
    PropertiesService.getScriptProperties()
      .setProperty('properties_data', JSON.stringify(body.data));
    Logger.log('物件データ更新: ' + Object.keys(body.data).length + '件');
    return ContentService.createTextOutput('OK');
  }

  // ── LINE Webhook ─────────────────────────────────────────────────
  var signature = '';
  if (e.headers) {
    signature = e.headers['X-Line-Signature'] || e.headers['x-line-signature'] || '';
  }

  // ① LSTEPにそのまま転送（LSTEPは今まで通り動く）
  forwardToLSTEP(bodyText, signature);

  // ② 物件番号だったら Push Message で物件カードを送信
  var events = body.events || [];
  for (var i = 0; i < events.length; i++) {
    var event = events[i];
    if (event.type !== 'message' || event.message.type !== 'text') continue;
    var text = (event.message.text || '').trim();
    var match = text.match(/^(?:物件)?(\d{2,4})$/);
    if (!match) continue;
    var propNum = ('000' + parseInt(match[1])).slice(-3);
    var prop = findProperty(propNum);
    if (prop) sendFlexMessage(event.source.userId, propNum, prop);
  }

  return ContentService.createTextOutput('OK');
}

// ================================================================
// LSTEP 転送
// ================================================================
function forwardToLSTEP(body, signature) {
  try {
    UrlFetchApp.fetch(LSTEP_WEBHOOK_URL, {
      method: 'post',
      payload: body,
      headers: { 'Content-Type': 'application/json', 'X-Line-Signature': signature },
      muteHttpExceptions: true
    });
  } catch (err) {
    Logger.log('LSTEP転送失敗: ' + err.message);
  }
}

// ================================================================
// 物件データ検索（Script Properties に保存済みのJSONを参照）
// ================================================================
function findProperty(propNum) {
  var raw = PropertiesService.getScriptProperties().getProperty('properties_data');
  if (!raw) return null;
  try {
    var data = JSON.parse(raw);
    return data[propNum] || data[String(parseInt(propNum))] || null;
  } catch (err) { return null; }
}

// ================================================================
// LINE Flex Message 送信（Push API）
// ================================================================
function sendFlexMessage(userId, propNum, prop) {
  var title    = (prop.title    || ('物件' + propNum)).substring(0, 40);
  var price    = prop.price    || '---';
  var layout   = prop.layout   || '---';
  var station  = (prop.station || '---').substring(0, 20);
  var detailUrl = prop.detail_url || '';
  var features = (prop.features || []).slice(0, 3).join('　');

  var bodyContents = [
    {type: 'text', text: '📍 物件' + propNum, weight: 'bold', size: 'md', color: '#dc3c1e'},
    {type: 'text', text: title, weight: 'bold', size: 'lg', wrap: true, margin: 'sm'},
    {
      type: 'box', layout: 'vertical', margin: 'md', spacing: 'xs',
      contents: [
        {type: 'box', layout: 'baseline', spacing: 'sm', contents: [
          {type: 'text', text: '💰 家賃', size: 'sm', color: '#888888', flex: 2},
          {type: 'text', text: price,    size: 'sm', flex: 3}
        ]},
        {type: 'box', layout: 'baseline', spacing: 'sm', contents: [
          {type: 'text', text: '🏠 間取', size: 'sm', color: '#888888', flex: 2},
          {type: 'text', text: layout,   size: 'sm', flex: 3}
        ]},
        {type: 'box', layout: 'baseline', spacing: 'sm', contents: [
          {type: 'text', text: '🚉 駅',  size: 'sm', color: '#888888', flex: 2},
          {type: 'text', text: station,  size: 'sm', flex: 3, wrap: true}
        ]}
      ]
    }
  ];

  if (features) {
    bodyContents.push({
      type: 'text', text: features, size: 'xs', color: '#888888', margin: 'md', wrap: true
    });
  }

  var bubble = {
    type: 'bubble',
    body: {type: 'box', layout: 'vertical', spacing: 'sm', contents: bodyContents}
  };

  if (detailUrl) {
    bubble.footer = {
      type: 'box', layout: 'vertical',
      contents: [{
        type: 'button', style: 'primary', color: '#dc3c1e',
        action: {type: 'uri', label: '📷 写真と詳細を見る', uri: detailUrl}
      }]
    };
  }

  UrlFetchApp.fetch('https://api.line.me/v2/bot/message/push', {
    method: 'post',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ' + LINE_CHANNEL_ACCESS_TOKEN
    },
    payload: JSON.stringify({
      to: userId,
      messages: [{
        type: 'flex',
        altText: '物件' + propNum + 'の情報をお届けします',
        contents: {type: 'carousel', contents: [bubble]}
      }]
    }),
    muteHttpExceptions: true
  });
}
