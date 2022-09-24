# def Tile_Creator_Circle(dim, params):
#      # Tensore delle distanze 
#     image = torch.zeros((3,dim,dim))
#     for i in range(dim):
#         for j in range(dim):
#             image[:,i,j] = torch.sqrt(torch.tensor((i-(dim-1)/2)**2 + (j-(dim-1)/2)**2)) - (dim/2)*params[0]
#     coeff = image.sigmoid()

#     # Creazione dei tensori per i colori 
#     color1_image = params[1].unsqueeze(-1).unsqueeze(-1)
#     color1_image = color1_image.expand(-1, dim, dim)

#     color2_image = params[2].unsqueeze(-1).unsqueeze(-1)
#     color2_image = color2_image.expand(-1, dim, dim)

#     return coeff*color1_image + (1-coeff)*color2_image, color1_image

# def Params_Creator_Circle():
#     a = torch.tensor(0.50)
#     a.requires_grad_(True)
#     color1 = torch.tensor([0.5,0.5,0.5])
#     color1.requires_grad_(True)
#     color2 = torch.tensor([0.5,0.5,0.5])
#     color2.requires_grad_(True)

#     params = [a, color1, color2]
#     return params

# def Tile_Creator_Ellipse(dim, params):
#      # Tensore delle distanze 
#     image = torch.zeros((3,dim,dim))
#     for i in range(dim):
#         for j in range(dim):
#             image[:,i,j] = ((i-(dim-1)/2)*(dim/2)*params[1])**2 + ((j-(dim-1)/2)*(dim/2)*params[0])**2-((dim/2)*params[0]*(dim/2)*params[1])**2
#     coeff = image.sigmoid()

#     # Creazione dei tensori per i colori 
#     color1_image = params[2].unsqueeze(-1).unsqueeze(-1)
#     color1_image = color1_image.expand(-1, dim, dim)

#     color2_image = params[3].unsqueeze(-1).unsqueeze(-1)
#     color2_image = color2_image.expand(-1, dim, dim)

#     return coeff*color1_image + (1-coeff)*color2_image, color1_image

# def Params_Creator_Ellipse():
#     a = torch.tensor(0.50)
#     a.requires_grad_(True)
#     b = torch.tensor(0.50)
#     b.requires_grad_(True)
#     color1 = torch.tensor([0.5,0.5,0.5])
#     color1.requires_grad_(True)
#     color2 = torch.tensor([0.5,0.5,0.5])
#     color2.requires_grad_(True)

#     params = [a, b, color1, color2]
#     return params

# def Tile_Creator_Square(dim, params):
#      # Tensore delle distanze 
#     image = torch.zeros((3,dim,dim))
#     for i in range(dim):
#         for j in range(dim):
#             image[:,i,j] = torch.max(torch.abs(torch.tensor(i-(dim-1)/2)/params[0]),torch.abs(torch.tensor(j-(dim-1)/2))/params[0]) - dim/2
#     coeff = image.sigmoid()

#     # Creazione dei tensori per i colori 
#     color1_image = params[1].unsqueeze(-1).unsqueeze(-1)
#     color1_image = color1_image.expand(-1, dim, dim)

#     color2_image = params[2].unsqueeze(-1).unsqueeze(-1)
#     color2_image = color2_image.expand(-1, dim, dim)

#     return coeff*color1_image + (1-coeff)*color2_image, color1_image

# def Params_Creator_Square():
#     a = torch.tensor(0.50)
#     a.requires_grad_(True)
#     color1 = torch.tensor([0.5,0.5,0.5])
#     color1.requires_grad_(True)
#     color2 = torch.tensor([0.5,0.5,0.5])
#     color2.requires_grad_(True)

#     params = [a, color1, color2]
#     return params

# def Tile_Creator_Rectangle(dim, params):
#      # Tensore delle distanze 
#     image = torch.zeros((3,dim,dim))
#     for i in range(dim):
#         for j in range(dim):
#             image[:,i,j] = torch.max(torch.abs(torch.tensor(i-(dim-1)/2)/params[0]),torch.abs(torch.tensor(j-(dim-1)/2))/params[1]) - dim/2
#     coeff = image.sigmoid()

#     # Creazione dei tensori per i colori 
#     color1_image = params[2].unsqueeze(-1).unsqueeze(-1)
#     color1_image = color1_image.expand(-1, dim, dim)

#     color2_image = params[3].unsqueeze(-1).unsqueeze(-1)
#     color2_image = color2_image.expand(-1, dim, dim)

#     return coeff*color1_image + (1-coeff)*color2_image, color1_image

# def Params_Creator_Rectangle():
#     a = torch.tensor(0.50)
#     a.requires_grad_(True)
#     b = torch.tensor(0.5)
#     b.requires_grad_(True)
#     color1 = torch.tensor([0.5,0.5,0.5])
#     color1.requires_grad_(True)
#     color2 = torch.tensor([0.5,0.5,0.5])
#     color2.requires_grad_(True)

#     params = [a, b, color1, color2]
#     return params

# def Tile_Creator_Triangle(dim, params):
#      # Tensore delle distanze 
#     image = torch.zeros((3,dim,dim))
#     for i in range(dim):
#         for j in range(dim):
#             image[:,i,j] = torch.max(torch.abs(-(i-(dim-1)/2)/params[1]),torch.abs(2*(j-(dim-1)/2)/params[0]) + (i-(dim-1)/2)/params[1]) - dim/2
#     coeff = image.sigmoid()

#     # Creazione dei tensori per i colori 
#     color1_image = params[2].unsqueeze(-1).unsqueeze(-1)
#     color1_image = color1_image.expand(-1, dim, dim)

#     color2_image = params[3].unsqueeze(-1).unsqueeze(-1)
#     color2_image = color2_image.expand(-1, dim, dim)

#     return coeff*color1_image + (1-coeff)*color2_image, color1_image

# def Params_Creator_Triangle():
#     a = torch.tensor(0.50)
#     a.requires_grad_(True)
#     b = torch.tensor(0.5)
#     b.requires_grad_(True)
#     color1 = torch.tensor([0.5,0.5,0.5])
#     color1.requires_grad_(True)
#     color2 = torch.tensor([0.5,0.5,0.5])
#     color2.requires_grad_(True)

#     params = [a, b, color1, color2]
#     return params
# def Tile_Creator_Trapezoid(dim, params):
#      # Tensore delle distanze 
#     image = torch.zeros((3,dim,dim))
#     for i in range(dim):
#         for j in range(dim):
#             image[:,i,j] = torch.max(torch.abs(2*(i-(dim-1)/2)/params[1]),torch.abs(3*(j-(dim-1)/2)/params[0]) + (i-(dim-1)/2)/params[1]) - dim
#     coeff = image.sigmoid()

#     # Creazione dei tensori per i colori 
#     color1_image = params[2].unsqueeze(-1).unsqueeze(-1)
#     color1_image = color1_image.expand(-1, dim, dim)

#     color2_image = params[3].unsqueeze(-1).unsqueeze(-1)
#     color2_image = color2_image.expand(-1, dim, dim)

#     return coeff*color1_image + (1-coeff)*color2_image, color1_image

# def Params_Creator_Trapezoid():
#     a = torch.tensor(0.50)
#     a.requires_grad_(True)
#     b = torch.tensor(0.5)
#     b.requires_grad_(True)
#     color1 = torch.tensor([0.5,0.5,0.5])
#     color1.requires_grad_(True)
#     color2 = torch.tensor([0.5,0.5,0.5])
#     color2.requires_grad_(True)

#     params = [a, b, color1, color2]
#     return params
