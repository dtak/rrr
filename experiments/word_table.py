import re

def word_tags(pred, pos, neg, maxwlen=5, maxwords=10, scale=None):
  if scale is None:
    if pos and neg:
      scale = max(max(p[0] for p in pos), -min(n[0] for n in neg))
    elif pos:
      scale = max(p[0] for p in pos)
    else:
      scale = -min(n[0] for n in neg)

  def smooth(ws):
    return (4./5.)*(0.25+ws)

  def style(weight):
    if pred == 'alt.atheism':
      weight = -weight
    if weight >= 0:
      color = 'rgba(0,255,0,{})'.format(min(1, smooth(weight/scale)))
    else:
      color = 'rgba(255,0,0,{})'.format(min(1, smooth(-weight/scale)))
    return 'margin-right: 5px; background: {}'.format(color)

  html = ""
  html += pred.split('.')[-1] + "<br>"

  wordlen = 0
  words = 0
  for weight, word in pos:
    html += "<span style='{}'>{}</span>".format(style(weight), word)
    wordlen += len(word)
    if wordlen > maxwlen:
      wordlen = 0
      html +="<br>"
    words += 1
    if words >= maxwords:
      break

  html += "<hr style='margin: 0; padding: 0; border-color: darkgrey;'>"

  wordlen = 0
  words = 0
  for weight, word in neg:
    html += "<span style='{}'>{}</span>".format(style(weight), word)
    wordlen += len(word)
    if wordlen > maxwlen:
      wordlen = 0
      html +="<br>"
    words += 1
    if words >= maxwords:
      break

  if not neg:
    html += "<span>&mdash;</span>"

  return html

def abs_weights_in(row):
  label, pos, neg = row
  poswts = [abs(wt) for wt, word in pos]
  negwts = [abs(wt) for wt, word in neg]
  return poswts + negwts

def max_weight_in(rows):
  wts = []
  for row in rows:
    wts += abs_weights_in(row)
  return max(wts)

def word_rows(columns, cutoff=None, startafter=None, **kwargs):
  scales = None
  if 'scale' not in kwargs:
    scales = [max_weight_in(rows) for rows in columns]
  html = ""
  for row in range(len(columns[0])):
    if startafter is not None and row < startafter:
      continue
    html += "<tr>"
    for col in range(len(columns)):
      html += "<td style='vertical-align: top'>"
      if scales:
        html += word_tags(*columns[col][row], scale=scales[col], **kwargs)
      else:
        html += word_tags(*columns[col][row], **kwargs)
      html += "</td>"
    html += "</tr>"
    if cutoff is not None and row >= cutoff:
      break
  return html

def word_table(columns, headers, cutoff=None, startafter=None, style='table-layout:fixed', **kwargs):
  html = "<table style='{}'><thead><tr>".format(style)
  for header in headers:
    html += "<th>{}</th>".format(header)
  html += "</tr></thead><tbody>"
  html += word_rows(columns, cutoff, startafter, **kwargs)
  html += "</tbody></table>"
  return html

def email_html(email, row, title=None):
  pred, poswts, negwts = row
  
  maxpos = max(wt for wt,_ in poswts) if poswts else 0
  maxneg = max(-wt for wt,_ in negwts) if negwts else 0
  maxwt = max(maxpos, maxneg)
  for wt, w in poswts:
    email = re.sub(r"([^a-zA-Z0-9]){}([^a-zA-Z0-9])".format(w),
                   "\\1{}\\2".format("<span style='background: rgba(0,255,0,{});'>{}</span>".format((wt/maxwt)**2, w)),
                   email)
  for wt, w in negwts:
    email = re.sub(r"([^a-zA-Z0-9]){}([^a-zA-Z0-9])".format(w),
                   "\\1{}\\2".format("<span style='background: rgba(255,0,0,{})'>{}</span>".format((-wt/maxwt)**2, w)),
                   email)
  email = email.replace('\n', '<br>')
  if title is None:
    title = 'Prediction: {}'.format(pred)
  else:
    title = '{} ({})'.format(title, pred)
  email = '<h4>{}</h4><hr>'.format(title) + email
  return email

def compare_emails(email, grad_row, lime_row):
  gmail = email_html(email, grad_row, 'Input gradients')
  lmail = email_html(email, lime_row, 'LIME')
  graddiv = '<div style="width:49%">' + gmail + '</div>'
  limediv = '<div style="width:49%;float:right;margin-left:2%">' + lmail + '</div>'
  return limediv + graddiv

