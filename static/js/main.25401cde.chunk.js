(this.webpackJsonpfront=this.webpackJsonpfront||[]).push([[0],{22:function(e,t,n){},30:function(e,t,n){"use strict";n.r(t);var c=n(1),i=n(14),r=n.n(i),a=(n(22),n(0)),s=function(e){var t=e.SideFunction;return Object(a.jsxs)("div",{children:[Object(a.jsxs)("nav",{className:"w3-sidebar w3-collapse w3-white w3-animate-left",style:{zIndex:"3",width:"300px"},id:"mySidebar",children:[Object(a.jsx)("br",{}),Object(a.jsxs)("div",{className:"w3-container",children:[Object(a.jsx)("span",{onClick:function(){return t.side_close()},className:"w3-hide-large w3-right w3-jumbo w3-padding w3-hover-grey",title:"close menu",children:Object(a.jsx)("i",{className:"fa fa-remove"})}),Object(a.jsx)("img",{alt:"myphoto",src:"a.jpg",style:{width:"45%"},className:"w3-round"}),Object(a.jsx)("br",{}),Object(a.jsx)("br",{}),Object(a.jsx)("h4",{children:Object(a.jsx)("b",{children:"DONCE's WEB"})})]}),Object(a.jsxs)("div",{className:"w3-bar-block",children:[Object(a.jsxs)("a",{href:"/#",onClick:function(){return t.side_close()},className:"w3-bar-item w3-button w3-padding w3-text-teal",children:[Object(a.jsx)("i",{className:"fa fa-th-large fa-fw w3-margin-right"}),"HOME"]}),Object(a.jsxs)("a",{href:"#about",onClick:function(){return t.side_close()},className:"w3-bar-item w3-button w3-padding",children:[Object(a.jsx)("i",{className:"fa fa-user fa-fw w3-margin-right"}),"ABOUT"]}),Object(a.jsxs)("a",{href:"#contact",onClick:function(){return t.side_close()},className:"w3-bar-item w3-button w3-padding",children:[Object(a.jsx)("i",{className:"fa fa-envelope fa-fw w3-margin-right"}),"CONTACT"]})]}),Object(a.jsx)("div",{className:"w3-panel w3-large",children:Object(a.jsx)("a",{style:{color:"black"},href:"https://github.com/doncechenjun/",children:Object(a.jsx)("i",{style:{margin:"auto"},className:"fa fa-github w3-hover-opacity"})})})]}),Object(a.jsx)("div",{className:"w3-overlay w3-hide-large w3-animate-opacity",onClick:function(){return t.side_close()},style:{cursor:"pointer"},title:"close side menu",id:"myOverlay"})]})},o=function(e){var t=e.SideFunction;return Object(a.jsxs)("header",{id:"portfolio",children:[Object(a.jsx)("a",{href:"/#",children:Object(a.jsx)("img",{alt:"myPhoto",src:"a.jpg",style:{width:"65px"},className:"w3-circle w3-right w3-margin w3-hide-large w3-hover-opacity"})}),Object(a.jsx)("span",{className:"w3-button w3-hide-large w3-xxlarge w3-hover-text-grey",onClick:function(){t.side_open()},children:Object(a.jsx)("i",{className:"fa fa-bars"})})]})},l=function(){return Object(a.jsxs)("footer",{className:"w3-container w3-padding-32 w3-dark-grey",children:[Object(a.jsxs)("div",{className:"w3-row-padding",children:[Object(a.jsxs)("div",{className:"w3-third",children:[Object(a.jsx)("h3",{children:"Starring my repository"}),Object(a.jsxs)("p",{children:["Give me a star on ",Object(a.jsx)("a",{style:{color:"blue"},href:"https://github.com/doncechenjun/",children:"GitHub"})," if you like my project!"]})]}),Object(a.jsx)("div",{className:"w3-third"}),Object(a.jsx)("div",{className:"w3-third"})]}),Object(a.jsx)("p",{children:"Powered by React"})]})},d=function(e){var t={side_open:function(){document.getElementById("mySidebar").style.display="block",document.getElementById("myOverlay").style.display="block"},side_close:function(){document.getElementById("mySidebar").style.display="none",document.getElementById("myOverlay").style.display="none"}};return Object(a.jsxs)("div",{children:[Object(a.jsx)(s,{SideFunction:t}),Object(a.jsxs)("div",{className:"w3-main",style:{marginLeft:"300px"},children:[Object(a.jsx)(o,{SideFunction:t}),e.children,Object(a.jsxs)("div",{className:"w3-container w3-padding-large",style:{marginBottom:"32px"},children:[Object(a.jsx)("h4",{children:Object(a.jsx)("b",{children:"About Me"})}),Object(a.jsx)("p",{children:"I specialize in material science, interest in programming and hardware designing."}),Object(a.jsx)("hr",{}),Object(a.jsx)(l,{})]})]})]})},j=n(12),h=function(e){var t=e.Article;return Object(a.jsx)("div",{className:"w3-third w3-container w3-margin-bottom",children:Object(a.jsxs)("a",{href:"#/Article/"+t.index,style:{textDecoration:"none"},children:[Object(a.jsx)("img",{src:t.imgPath,alt:t.date,style:{width:"100%"},className:"w3-hover-opacity"}),Object(a.jsxs)("div",{style:{height:"7rem",overflow:"hidden"},className:"w3-container w3-white",children:[Object(a.jsx)("p",{children:Object(a.jsx)("b",{children:t.title})}),Object(a.jsx)("p",{children:t.intro})]})]})})},b=function(e){var t=e.ArticleList;return Object(a.jsx)("div",{children:Object(a.jsx)("div",{className:"w3-row-padding",children:t.map((function(e){return Object(a.jsx)(h,{Article:e},e.index)}))})})},u=function(e){var t=e.CurrentPage,n=e.totalPage,c=e.setPage,i=[];if(n<8)for(var r=1;r<n+1;r++)i.push(r);else t>n-4?i.push(1,"...",n-4,n-3,n-2,n-1,n):t<5?i.push(1,2,3,4,5,"...",n):i.push(1,"...",t-1,t,t+1,"...",n);return Object(a.jsx)("div",{className:"w3-center w3-padding-32",children:Object(a.jsxs)("div",{className:"w3-container",children:[Object(a.jsx)("button",{className:" w3-button w3-white w3-round",onClick:function(){t>1&&c(t-1)},children:"\xab"}),i.map((function(e){return t===e?Object(a.jsx)("button",{className:" w3-button w3-black w3-round",children:e},e):Object(a.jsx)("button",{className:" w3-button w3-white w3-round",onClick:function(){c(e)},children:e},e)})),Object(a.jsx)("button",{className:" w3-button w3-white w3-round",onClick:function(){t<n&&c(t+1)},children:"\xbb"})]})})},m=[{index:"0",title:"markdown file component",date:"2021-04-03",topic:"Others",imgPath:"/articles/20210403/img.png",articlePath:"/articles/20210403/article.txt",intro:"Using Markdown file to write my blog."},{index:"1",title:"React common used component",date:"2021-05-05",topic:"Programming",imgPath:"/articles/20210505/img.jpg",articlePath:"/articles/20210505/article.txt",intro:"Some common used component include pagination, rightclick menu ...etc."}],w=function(){var e=Object(c.useState)(m),t=Object(j.a)(e,2),n=t[0],i=t[1],r=Object(c.useState)(1),s=Object(j.a)(r,2),o=s[0],l=s[1],d=Math.floor(n.length/6)+1,h=function(e,t){e.target.parentNode.childNodes.forEach((function(e){return e.style.backgroundColor="transparent"})),e.target.style.backgroundColor="#ccc";for(var n=[],c=0;c<m.length;c++)t===m[c].topic&&n.push(m[c]);i(n)},w=function(e){i(m),e.target.parentNode.childNodes.forEach((function(e){return e.style.backgroundColor="transparent"})),e.target.style.backgroundColor="#ccc"},g=function(e){return n.slice(-6*e,n.length-6*(e-1)).reverse()};return Object(a.jsxs)("div",{children:[Object(a.jsx)("div",{className:"w3-container",children:Object(a.jsxs)("div",{className:"w3-section w3-bottombar w3-padding-16",children:[Object(a.jsx)("span",{className:"w3-margin-right",children:"topics:"}),Object(a.jsx)("button",{id:"btn_all",className:"w3-button",style:{backgroundColor:"#ccc"},onClick:function(e){w(e)},children:"ALL"}),Object(a.jsx)("button",{id:"btn_Programming",className:"w3-button",onClick:function(e){h(e,"Programming")},children:"Programming"}),Object(a.jsx)("button",{id:"btn_Hardware",className:"w3-button",onClick:function(e){h(e,"Hardware")},children:"Hardware"}),Object(a.jsx)("button",{id:"btn_Others",className:"w3-button",onClick:function(e){h(e,"Others")},children:"Others"})]})}),Object(a.jsx)(b,{ArticleList:g(o)}),Object(a.jsx)(u,{CurrentPage:o,totalPage:d,setPage:function(e){return l(e)}})]})};var g={width:"100%",padding:"2vh 3vw"},x=function(e){return Object(c.useEffect)((function(){var t=document.getElementById("showfile");fetch(m[e.match.params.id].articlePath).then((function(e){return e.text()})).then((function(e){t.innerHTML=function(e){var t="";function n(e){return new Option(e).innerHTML}function c(e){return n(e).replace(/!\[([^\]]*)]\(([^(]+)\)/g,'<img alt="$1" src="$2">').replace(/\[([^\]]+)]\(([^(]+?)\)/g,"$1".link("$2")).replace(/`([^`]+)`/g,"<code>$1</code>").replace(/(\*\*|__)(?=\S)([^\r]*?\S[*_]*)\1/g,"<strong>$2</strong>").replace(/(\*|_)(?=\S)([^\r]*?\S)\1/g,"<em>$2</em>")}return e.replace(/^\s+|\r|\s+$/g,"").replace(/\t/g,"    ").split(/\n\n+/).forEach((function(e,i,r){i=e[0],t+=(r={"*":[/\n\* /,"<ul><li>","</li></ul>"],1:[/\n[1-9]\d*\.? /,"<ol><li>","</li></ol>"]," ":[/\n {4}/,"<pre><code>","</code></pre>","\n"],">":[/\n> /,"<blockquote>","</blockquote>","\n"]}[i])?r[1]+("\n"+e).split(r[0]).slice(1).map(r[3]?n:c).join(r[3]||"</li>\n<li>")+r[2]:"#"===i?"<h"+(i=e.indexOf(" "))+">"+c(e.slice(i+1))+"</h"+i+">":"<"===i?e:"<p>"+c(e)+"</p>"})),t}(e)}))})),Object(a.jsxs)("div",{style:g,children:[Object(a.jsx)("h1",{children:m[e.match.params.id].title}),Object(a.jsx)("p",{style:{color:"#aaa"},children:m[e.match.params.id].date}),Object(a.jsx)("div",{id:"showfile",style:{padding:"2vh 2vw",backgroundColor:"white"}})]})},p=function(){return Object(a.jsxs)("div",{style:{padding:"2vh 3vw"},children:[Object(a.jsx)("h1",{children:"About"}),Object(a.jsx)("br",{}),Object(a.jsx)("p",{children:"I build this website  not only due to have a record of my own projects but also want to assist somebody who can find the conclusion of his/her problem here."})]})},O=function(){return Object(a.jsxs)("div",{style:{padding:"2vh 3vw"},children:[Object(a.jsx)("h1",{children:"Contact"}),Object(a.jsx)("iframe",{src:"https://docs.google.com/forms/d/e/1FAIpQLSeyocRpR277ZE2K9nqPCxj8vWFjfZguTdy2axmrWDVgBZvMXw/viewform?embedded=true",width:"640",height:"600",frameborder:"0",marginheight:"0",marginwidth:"0",children:"\u8f09\u5165\u4e2d\u2026"})]})},f=function(e){e&&e instanceof Function&&n.e(3).then(n.bind(null,31)).then((function(t){var n=t.getCLS,c=t.getFID,i=t.getFCP,r=t.getLCP,a=t.getTTFB;n(e),c(e),i(e),r(e),a(e)}))},v=(n(24),n(15)),y=n(2);r.a.render(Object(a.jsx)(v.a,{children:Object(a.jsx)(y.c,{children:Object(a.jsxs)(d,{children:[Object(a.jsx)(y.a,{exact:!0,path:"/",component:w}),Object(a.jsx)(y.a,{path:"/Article/:id",component:x}),Object(a.jsx)(y.a,{path:"/About",component:p}),Object(a.jsx)(y.a,{path:"/Contact",component:O})]})})}),document.getElementById("root")),f()}},[[30,1,2]]]);
//# sourceMappingURL=main.25401cde.chunk.js.map