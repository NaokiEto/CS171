OBJECTS=main.o readpng.o
CFLAGS=-g
LDFLAGS= -L/usr/lib/nvidia -L/usr/X11R6/lib -lGL -lGLU -lglut -lpng #-lXi -lXmu

glslRenderer: $(OBJECTS)
	g++ $(CFLAGS) -o glslRenderer $^ $(LDFLAGS)

.cpp.o:
	g++ $(CFLAGS) -c $<

clean:
	-rm $(OBJECTS) glslRenderer
