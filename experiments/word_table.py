def word_tags(pred, pos, neg, maxwlen=5, maxwords=10, scale=None):
  if scale is None:
    if pos and neg:
      scale = max(max(p[0] for p in pos), -min(n[0] for n in neg))
    elif pos:
      scale = max(p[0] for p in pos)
    else:
      scale = -min(n[0] for n in neg)
    
  def style(weight):
    if pred == 'alt.atheism':
      weight = -weight
    if weight >= 0:
      color = 'rgba(0,255,0,{})'.format(min(1, weight/scale))
    else:
      color = 'rgba(255,0,0,{})'.format(min(1, -weight/scale))
    return 'margin-right: 5px; background: {}'.format(color)
  
  html = ""
  html += pred + "<br>"
  
  wordlen = 0
  words = 0
  for weight, word in pos:
    html += "<span style='{}'>{}</span>".format(style(weight), word)

    wordlen += len(word)
    if wordlen > maxwlen:
      wordlen = 0
      html +="<br>"
    words += 1
    if words >= 10:
      break
    
  html += "<hr style='margin: 0; padding: 0'>"
  
  wordlen = 0
  words = 0
  for weight, word in neg:
    html += "<span style='{}'>{}</span>".format(style(weight), word)

    wordlen += len(word)
    if wordlen > maxwlen:
      wordlen = 0
      html +="<br>"
    words += 1
    if words >= 10:
      break
    
  if not neg:
    html += "<span>&mdash;</span>"
  return html
    
def word_rows(columns, cutoff=None, startafter=None, **kwargs):
  html = ""
  for row in range(len(columns[0])):
    if startafter is not None and row < startafter:
      continue
    html += "<tr>"
    for col in range(len(columns)):
      html += "<td style='vertical-align: top'>"
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
