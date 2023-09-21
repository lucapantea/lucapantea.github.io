---
title:
layout: default
permalink: /projects/
published: true
---
<p>

<div class="projectContainer">

  <div class="gallery">
  
    {% for project in site.projects %}
    
    {% if project.redirect %}
    <a href="{{ project.redirect }}" target="_blank" class="projectTileLink">
    {% else %}
    <a href="{{ project.url | prepend: site.baseurl | prepend: site.url }}" class="projectTileLink">
    {% endif %}
    
      <div class="projectTile">
      
        {% if project.image %}
        <div class="projectImage" style="background-image: url('{{ project.image }}');"></div>
        {% endif %}
        
        <div class="projectInfo">
          <h2>{{ project.title }}</h2>
          <p>{{ project.description }}</p>
        </div>
      
      </div>
    
    </a>
    
    {% endfor %}

  </div>

</div>
