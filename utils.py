import numpy as np
import pygame as pg


def display_text(text, font, color, center_x, center_y, display):
    """
        Function that displays text on the screen.
        text:       Text to display.
        font:       Font that the text fill use.
        color:      Color of the text to display.
        center_x:   Horizontal coordinate of the center of the text box.
        center_y:   Vertical coordinate of the center of the text box.
        display:    Screen surface that will display the text.
    """
    text_surf = font.render(text, True, color)
    text_rect = text_surf.get_rect()
    text_rect.center = center_x, center_y
    display.blit(text_surf, text_rect)
    

def update_button(msg, rect, ic, ac, tc, bc, font, display, mouse, bw=2, action=None, arg=None):
    """
        Function that displays an interactive button.
        msg:     Text inside the button.
        rect:    pygame.Rect object of the size of the desired button.
        ic:      Color of the button when the mouse is not hovering it.
        ac:      Color of the button when the mouse is hovering it.
        tc:      Color of the text inside the button.
        bc:      Color of the border of the button.
        font:    Font that the text inside the color will use (pygame.font.Font).
        display: Screen surface that will display the button.
        bw:      Width in pixels of the border of the button.
        action:  Method to be called when the button is pressed.
        arg:     Argument to be passed to the method determined by the action parameter.
    """
    click = pg.mouse.get_pressed()

    if rect.x + rect.w > mouse[0] > rect.x and rect.y + rect.h > mouse[1] > rect.y:
        pg.draw.rect(display, ac, rect)

        if click[0] == 1 and action is not None:
            if arg is not None:
                action(arg)
            else:
                action()
    else:
        pg.draw.rect(display, ic, rect)

    pg.draw.rect(display, bc, rect, bw)

    display_text(msg, font, tc, rect.x + (rect.w / 2), rect.y + (rect.h / 2), display)
    
    
# display the play grid
def display_grid(grid, screen, mouse, cell_margin, grid_w, grid_h, margin_x, margin_y):
    cells = [s for s, v in np.ndenumerate(grid)]
    for x in cells:
        color = (84, 145, 255)
        if grid[x] == 1:
            color = (255, 255, 255)  # white
        elif grid[x] == 2:
            color = (0, 0, 0)  # black
        elif (mouse[1] - margin_y) // (grid_h + cell_margin) == x[0] \
                and (mouse[0] - margin_x) // (grid_w + cell_margin) == x[1]:
            color = (105, 158, 255)
        pg.draw.rect(screen, color, [(cell_margin + grid_w) * x[1] + cell_margin + margin_x,
                                     (cell_margin + grid_h) * x[0] + cell_margin + margin_y, grid_w, grid_h])
